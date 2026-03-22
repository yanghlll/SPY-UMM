"""
Show-o2 Spy Game Wrapper.

Provides a unified interface for both image generation (t2i) and
image understanding (mmu) modes needed by the spy game.

Closely follows Show-o2's original API:
  - generate_images(): mirrors inference_t2i.py
  - judge_vote(): mirrors inference_mmu.py
  - compute_flow_loss(): mirrors train_stage_one.py's training loop
  - compute_voting_logprobs(): teacher-forced log probs for GRPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Dict, Any, Optional, Tuple
from PIL import Image


class Showo2SpyWrapper(nn.Module):
    """Wraps Showo2Qwen2_5 for the spy game.

    Provides:
    - generate_images(): Text-to-image generation for all players
    - compute_flow_loss(): Flow matching loss with per-sample reward weighting
    - judge_vote(): Image understanding mode for voting
    - compute_voting_logprobs(): Per-token log probs for GRPO
    """

    def __init__(self, model, vae_model, text_tokenizer, showo_token_ids,
                 transport, sampler, config):
        super().__init__()
        self.model = model
        self.vae = vae_model
        self.tokenizer = text_tokenizer
        self.token_ids = showo_token_ids
        self.transport = transport
        self.sampler = sampler
        self.config = config

        # Extract commonly used params (these should already reflect
        # the +1 adjustment from add_time_embeds done in train_spy_umm.py)
        preproc = config.dataset.preprocessing
        self.num_t2i_tokens = preproc.num_t2i_image_tokens
        self.num_mmu_tokens = preproc.num_mmu_image_tokens
        self.max_seq_len = preproc.max_seq_length
        self.image_latent_dim = config.model.showo.image_latent_dim
        self.latent_height = preproc.latent_height
        self.latent_width = preproc.latent_width
        self.patch_size = config.model.showo.patch_size
        self.add_time_embeds = config.model.showo.add_time_embeds

        # Token IDs
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.pad_id = text_tokenizer.pad_token_id
        self.img_pad_id = showo_token_ids['img_pad_id']

        # Max text length = max_seq_len - num_image_tokens - 4 special tokens
        # [bos, text, boi, image_tokens, eoi, eos, padding]
        self.max_text_len = self.max_seq_len - self.num_t2i_tokens - 4

        # 这里需要check一下
        # Precompute system prompt tokens for MMU
        # Matches inference_mmu.py lines 100-103
        self.sys_prompt_ids = text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>",
            add_special_tokens=False
        )['input_ids']
        self.role_a_ids = text_tokenizer(
            "\n<|im_start|>user\n", add_special_tokens=False
        )['input_ids']
        self.role_b_ids = text_tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )['input_ids']

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    # ==================== IMAGE GENERATION ====================

    @torch.no_grad()
    def generate_images(self, prompts: List[str],
                        guidance_scale: float = 5.0,
                        num_steps: int = 50) -> Dict[str, Any]:
        """Generate images from text prompts using Show-o2's t2i pipeline.

        Mirrors inference_t2i.py lines 130-212.

        Args:
            prompts: List of text descriptions.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of ODE sampling steps.

        Returns:
            Dict with 'images' (list of PIL Images), 'latents' (tensor).
        """
        from models.misc import prepare_gen_input
        from models import omni_attn_mask_naive
        from utils import denorm

        device = self.device
        dtype = self.dtype

        batch_text_tokens, batch_text_tokens_null, batch_mod_pos, batch_mod_pos_null = \
            prepare_gen_input(
                prompts, self.tokenizer, self.num_t2i_tokens,
                self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
                self.pad_id, self.img_pad_id, self.max_text_len, device
            )

        # Initial noise: [B, C, H*p, W*p]
        z = torch.randn(
            len(prompts), self.image_latent_dim,
            self.latent_height * self.patch_size,
            self.latent_width * self.patch_size
        ).to(dtype).to(device)

        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_mod_pos, batch_mod_pos_null], dim=0)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_mod_pos

        # Attention mask: matches train_stage_one.py line 429-432
        block_mask = omni_attn_mask_naive(
            text_tokens.size(0), self.max_seq_len,
            modality_positions, device
        ).to(dtype)

        # model_kwargs matches inference_t2i.py lines 162-170
        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=self.max_seq_len,
            guidance_scale=guidance_scale,
        )

        # ODE sampling: matches inference_t2i.py lines 172-181
        sample_fn = self.sampler.sample_ode(
            sampling_method=self.config.transport.sampling_method,
            num_steps=num_steps,
            atol=self.config.transport.atol,
            rtol=self.config.transport.rtol,
            reverse=self.config.transport.reverse,
            time_shifting_factor=self.config.transport.time_shifting_factor,
        )
        samples = sample_fn(z, self.model.t2i_generate, **model_kwargs)[-1]

        # Take conditional half if using guidance
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]

        # Decode through VAE: matches inference_t2i.py lines 183-188
        latents = samples
        samples_for_vae = samples.unsqueeze(2)  # Add temporal dim for WanVAE
        images_tensor = self.vae.batch_decode(samples_for_vae).squeeze(2)

        # Convert to PIL
        images_np = denorm(images_tensor)
        pil_images = [Image.fromarray(img) for img in images_np]

        return {
            'images': pil_images,
            'latents': latents,
            'images_tensor': images_tensor,
        }

    # ==================== FLOW MATCHING TRAINING ====================

    def compute_flow_loss(self, prompts: List[str],
                          target_latents: Optional[torch.Tensor] = None,
                          reward_weights: Optional[torch.Tensor] = None,
                          ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss with per-sample reward weighting.

        Mirrors train_stage_one.py's training loop (lines 345-451):
        1. Prepare text tokens and modality positions
        2. Encode target images to patchified labels
        3. Sample timestep t, create noised latent xt, compute target velocity ut
        4. Forward through model to get v_pred
        5. Compute per-sample MSE loss between v_pred and ut
        6. Weight by rewards

        Args:
            prompts: Text prompts for conditioning.
            target_latents: [B, C, H, W] target image latents.
                           If None, uses random noise as target.
            reward_weights: [B] reward weights per sample.

        Returns:
            Dict with 'loss', 'per_sample_loss', 'flow_loss_unweighted'.
        """
        from models.misc import prepare_gen_input
        from models import omni_attn_mask_naive

        device = self.device
        dtype = self.dtype
        B = len(prompts)
        p = self.patch_size
        c = self.image_latent_dim
        h_ = self.latent_height   # latent height in patches (e.g., 27)
        w_ = self.latent_width    # latent width in patches (e.g., 27)

        # Prepare text tokens: matches train_stage_one.py line 403 + prepare_gen_input
        batch_text_tokens, _, batch_mod_pos, _ = prepare_gen_input(
            prompts, self.tokenizer, self.num_t2i_tokens,
            self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
            self.pad_id, self.img_pad_id, self.max_text_len, device
        )

        # If no target latents, generate random ones
        if target_latents is None:
            target_latents = torch.randn(
                B, c, h_ * p, w_ * p
            ).to(dtype).to(device)

        target_latents = target_latents.to(dtype).to(device)

        # Sample timesteps and create noised versions
        # Matches train_stage_one.py prepare_latents_and_labels() lines 345-393
        t_list, xt_list, ut_list = [], [], []
        for i in range(B):
            # transport.sample returns (t, x0_noise, x1_data)
            t_i, x0_i, x1_i = self.transport.sample(target_latents[i:i + 1])
            # Move t to correct device (transport creates t on CPU)
            t_i = t_i.to(device)
            # path_sampler.plan returns (t, xt_noised, ut_velocity_target)
            t_plan, xt_i, ut_i = self.transport.path_sampler.plan(t_i, x0_i, x1_i)
            t_list.append(t_plan.to(device))
            xt_list.append(xt_i.to(device))
            ut_list.append(ut_i.to(device))

        t = torch.stack(t_list, dim=0).squeeze(-1)  # [B]
        xt = torch.cat(xt_list, dim=0)   # [B, C, H, W]
        ut = torch.cat(ut_list, dim=0)   # [B, C, H, W]

        # Create attention mask
        block_mask = omni_attn_mask_naive(
            B, self.max_seq_len, batch_mod_pos, device
        ).to(dtype)

        # Create image masks: must be [B, max_seq_len] (full sequence length).
        # Model internally expands to [B, max_seq_len, p*p*c] and masks
        # the time embed position. We set 1s at image token positions.
        image_masks = torch.zeros(B, self.max_seq_len, device=device)
        for i, mod_pos in enumerate(batch_mod_pos):
            for offset, length in mod_pos:
                if length > 0:
                    image_masks[i, offset:offset + length] = 1

        # Forward pass through model
        # The model.forward() with text_labels=None, image_labels=not None
        # returns (logits, loss_flow) — 2 values, NOT 3!
        # See modeling_showo2_qwen2_5.py lines 401-403
        logits, loss_flow = self.model(
            text_tokens=batch_text_tokens,
            image_latents=xt.to(dtype),
            t=t.to(dtype),
            attention_mask=block_mask,
            text_masks=torch.ones_like(batch_text_tokens, dtype=torch.bool),
            image_masks=image_masks,
            text_labels=None,   # No NTP loss in generation phase
            image_labels=ut.to(dtype),  # velocity target, model will patchify internally
            modality_positions=batch_mod_pos,
            output_hidden_states=True,
            max_seq_len=self.max_seq_len,
        )

        # For per-sample reward weighting, we need per-sample losses.
        # model.forward() returns scalar mean loss. We recompute per-sample
        # by doing another forward pass without labels to get v_pred,
        # then manually compute per-sample MSE.
        per_sample_loss = self._compute_per_sample_flow_loss(
            batch_text_tokens, xt, t, ut,
            block_mask, image_masks, batch_mod_pos,
        )

        # Apply reward weighting
        if reward_weights is not None:
            reward_weights = reward_weights.to(device).to(dtype)
            weighted_loss = (reward_weights * per_sample_loss).sum() / (
                reward_weights.sum().clamp(min=1e-8)
            )
        else:
            weighted_loss = per_sample_loss.mean()

        return {
            'loss': weighted_loss,
            'flow_loss_unweighted': loss_flow.detach(),
            'per_sample_loss': per_sample_loss.detach(),
        }

    def _compute_per_sample_flow_loss(
        self,
        text_tokens: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        block_mask: torch.Tensor,
        image_masks: torch.Tensor,
        modality_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample MSE flow matching loss.

        Does a forward pass WITHOUT labels to get v_pred (velocity prediction),
        then computes per-sample MSE against the ground-truth velocity ut.

        Mirrors the flow of modeling_showo2_qwen2_5.py forward() lines 409-433
        (the no-labels branch that returns v_pred).

        Args:
            text_tokens: [B, seq_len] text token IDs.
            xt: [B, C, H, W] noised image latents.
            t: [B] or [B, 1] timesteps.
            ut: [B, C, H, W] ground-truth velocity targets.
            block_mask: Attention mask.
            image_masks: [B, num_tokens] image token mask.
            modality_positions: [B, num_images, 2] modality position info.

        Returns:
            [B] tensor of per-sample flow matching losses.
        """
        dtype = self.dtype
        B = text_tokens.size(0)

        # Forward without labels → returns (logits, v_pred)
        # v_pred shape: [B, C, H, W] (already unpatchified by the model)
        # See modeling_showo2_qwen2_5.py lines 409-465
        _, v_pred = self.model(
            text_tokens=text_tokens,
            image_latents=xt.to(dtype),
            t=t.to(dtype),
            attention_mask=block_mask,
            text_masks=None,
            image_masks=image_masks,
            text_labels=None,
            image_labels=None,   # No labels → returns v_pred
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=self.max_seq_len,
        )

        # v_pred: [B, C, H, W], ut: [B, C, H, W]
        # Compute per-sample MSE: mean over (C, H, W) dimensions
        per_sample_mse = F.mse_loss(
            v_pred.float(), ut.float(), reduction='none'
        ).mean(dim=[1, 2, 3])  # [B]

        return per_sample_mse.to(dtype)

    # ==================== IMAGE UNDERSTANDING (VOTING) ====================

    def _encode_image_for_mmu(self, image_latents: torch.Tensor) -> torch.Tensor:
        """Encode image latents through dual-path embedder for understanding.

        Mirrors inference_mmu.py lines 117-121.

        Args:
            image_latents: [1, C, H, W] image latents from VAE.

        Returns:
            [1, L, hidden_size] fused image embeddings.
        """
        dtype = self.dtype
        if len(image_latents.shape) == 3:
            image_latents = image_latents.unsqueeze(0)

        image_embeds_und = self.model.image_embedder_und(image_latents.to(dtype))
        image_embeds_gen = self.model.image_embedder_gen(image_latents.to(dtype))
        image_embeds_und = image_embeds_und + self.model.position_embedding(
            self.model.image_position_ids
        )
        image_embeds_und = self.model.und_trans(image_embeds_und)['last_hidden_state']
        image_embeds = self.model.fusion_proj(
            torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
        )
        return image_embeds

    def _build_mmu_input(self, image_embeds: torch.Tensor,
                         question: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Build input embeddings and attention mask for MMU mode.

        Supports variable-length image embeddings (e.g. concatenated
        multi-image embeddings for grid voting).

        Args:
            image_embeds: [1, L, hidden_size] fused image embeddings.
                          L can be num_mmu_tokens (single image) or
                          N * num_mmu_tokens (multi-image grid).
            question: Text question/prompt.

        Returns:
            Tuple of (input_embeds, attention_mask, prompt_len).
        """
        from models import omni_attn_mask_naive

        device = self.device
        dtype = self.dtype
        num_img_tokens = image_embeds.shape[1]  # actual image token count

        input_ids = self.tokenizer(question, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor(
            [self.bos_id] + self.sys_prompt_ids + self.role_a_ids
        ).to(device)[None, :]
        text_tokens_b = torch.tensor(
            [self.boi_id, self.eoi_id] + input_ids + self.role_b_ids
        ).to(device)[None, :]

        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b)

        if self.add_time_embeds:
            time_embeds = self.model.time_embed(
                torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype
            )
            if hasattr(self.model, 'time_embed_proj'):
                time_embeds = self.model.time_embed_proj(time_embeds)
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],      # [boi]
                time_embeds,
                image_embeds,              # [1, num_img_tokens, hidden]
                text_embeds_b[:, 1:],      # [eoi] + question + role_b
            ], dim=1).to(dtype)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, num_img_tokens]
            )[None, None, :].to(device)
        else:
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],
                image_embeds,
                text_embeds_b[:, 1:],
            ], dim=1).to(dtype)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, num_img_tokens]
            )[None, None, :].to(device)

        attention_mask = omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=device, inverted=True,
        ).to(dtype)

        prompt_len = input_embeds.shape[1]
        return input_embeds, attention_mask, prompt_len

    @torch.no_grad()
    def judge_vote(self, image_latents_list: List[torch.Tensor],
                   question: str,
                   max_new_tokens: int = 512,
                   temperature: float = 1.0,
                   top_k: int = 50) -> str:
        """Generate vote text using Show-o2's understanding mode.

        Like Vision-Zero: the voter sees ALL players' images as a labeled
        grid, then decides which player is the spy.

        Args:
            image_latents_list: List of [1, C, H, W] latents for each player's image.
            question: Voting prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            Generated text response.
        """
        if len(image_latents_list) == 1:
            # Single image: use directly
            image_latents = image_latents_list[0]
        else:
            # Multiple images: create a grid image (all players visible),
            # then re-encode as a single image. This keeps the same token
            # count as a single image while showing all players' work.
            from utils import denorm
            pil_images = []
            for lat in image_latents_list:
                img_tensor = self.vae.batch_decode(lat.unsqueeze(2)).squeeze(2)
                img_np = denorm(img_tensor)
                pil_images.append(Image.fromarray(img_np[0]))

            # Build labeled grid
            N = len(pil_images)
            cols = min(N, 2)
            rows = (N + cols - 1) // cols
            cell_size = pil_images[0].size[0] // cols  # shrink to fit
            grid_size = pil_images[0].size[0]  # same total size as single image
            grid = Image.new('RGB', (grid_size, grid_size), (200, 200, 200))
            for idx, img in enumerate(pil_images):
                r, c = divmod(idx, cols)
                x, y = c * cell_size, r * cell_size
                grid.paste(img.resize((cell_size, cell_size)), (x, y))

            # Re-encode grid as single latent (same shape as one image)
            import torchvision.transforms as T
            grid_tensor = T.ToTensor()(grid).unsqueeze(0).to(self.device)
            grid_tensor = grid_tensor * 2.0 - 1.0
            image_latents = self.vae.sample(
                grid_tensor.unsqueeze(2).to(self.dtype)
            ).squeeze(2)

        image_embeds = self._encode_image_for_mmu(image_latents)
        input_embeds, attention_mask, _ = self._build_mmu_input(image_embeds, question)

        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            eos_token=self.tokenizer.eos_token_id,
        )

        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return text[0]

    # ==================== FLOW-GRPO SUPPORT ====================

    def make_velocity_fn(self, text_tokens: torch.Tensor,
                         block_mask: torch.Tensor,
                         image_masks: torch.Tensor,
                         modality_positions: torch.Tensor,
                         guidance_scale: float = 0.0,
                         text_tokens_null: Optional[torch.Tensor] = None,
                         block_mask_null: Optional[torch.Tensor] = None,
                         modality_positions_null: Optional[torch.Tensor] = None,
                         ) -> Callable:
        """Create a velocity prediction function for Flow-GRPO.

        Returns a callable (x_t, t) -> v_theta that wraps the Show-o2 model's
        forward pass for use with FlowGRPO's SDE sampling.

        Args:
            text_tokens: [B, seq_len] text token IDs (conditional).
            block_mask: Attention mask (conditional).
            image_masks: [B, num_tokens] image token mask.
            modality_positions: Modality position info (conditional).
            guidance_scale: CFG scale. If > 0, requires null inputs.
            text_tokens_null: [B, seq_len] unconditional text tokens (for CFG).
            block_mask_null: Unconditional attention mask.
            modality_positions_null: Unconditional modality positions.

        Returns:
            Callable (x_t: [B,C,H,W], t: [B]) -> v_theta: [B,C,H,W]
        """
        dtype = self.dtype

        def velocity_fn(x_t, t, **kwargs):
            B_actual = x_t.shape[0]

            if guidance_scale > 0 and text_tokens_null is not None:
                # CFG: concatenate conditional and unconditional inputs
                x_t_both = torch.cat([x_t, x_t], dim=0)
                t_both = torch.cat([t, t], dim=0)
                text_both = torch.cat([text_tokens, text_tokens_null], dim=0)
                mask_both = torch.cat([block_mask, block_mask_null], dim=0)
                mod_both = torch.cat([modality_positions, modality_positions_null], dim=0)
                img_masks_both = torch.cat([image_masks, image_masks], dim=0)

                _, v_pred = self.model(
                    text_tokens=text_both,
                    image_latents=x_t_both.to(dtype),
                    t=t_both.to(dtype),
                    attention_mask=mask_both,
                    text_masks=None,
                    image_masks=img_masks_both,
                    text_labels=None,
                    image_labels=None,
                    modality_positions=mod_both,
                    output_hidden_states=True,
                    max_seq_len=self.max_seq_len,
                )

                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                return v_guided
            else:
                _, v_pred = self.model(
                    text_tokens=text_tokens,
                    image_latents=x_t.to(dtype),
                    t=t.to(dtype),
                    attention_mask=block_mask,
                    text_masks=None,
                    image_masks=image_masks,
                    text_labels=None,
                    image_labels=None,
                    modality_positions=modality_positions,
                    output_hidden_states=True,
                    max_seq_len=self.max_seq_len,
                )
                return v_pred

        return velocity_fn

    def prepare_flow_grpo_inputs(self, prompts: List[str],
                                 guidance_scale: float = 0.0
                                 ) -> Dict[str, Any]:
        """Prepare all inputs needed for Flow-GRPO velocity function.

        Args:
            prompts: List of text prompts.
            guidance_scale: CFG scale.

        Returns:
            Dict with 'z_init', 'velocity_fn_kwargs', 'model_kwargs' for
            creating velocity_fn and initial noise.
        """
        from models.misc import prepare_gen_input
        from models import omni_attn_mask_naive

        device = self.device
        dtype = self.dtype
        B = len(prompts)

        batch_text_tokens, batch_text_tokens_null, batch_mod_pos, batch_mod_pos_null = \
            prepare_gen_input(
                prompts, self.tokenizer, self.num_t2i_tokens,
                self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
                self.pad_id, self.img_pad_id, self.max_text_len, device
            )

        # Initial noise
        z = torch.randn(
            B, self.image_latent_dim,
            self.latent_height * self.patch_size,
            self.latent_width * self.patch_size,
        ).to(dtype).to(device)

        # Attention mask
        block_mask = omni_attn_mask_naive(
            B, self.max_seq_len, batch_mod_pos, device
        ).to(dtype)

        # image_masks must be [B, max_seq_len] with 1s at image token positions,
        # matching compute_flow_loss() and the model's expected input shape.
        image_masks = torch.zeros(B, self.max_seq_len, device=device)
        for i, mod_pos in enumerate(batch_mod_pos):
            for offset, length in mod_pos:
                if length > 0:
                    image_masks[i, offset:offset + length] = 1

        velocity_fn_kwargs = dict(
            text_tokens=batch_text_tokens,
            block_mask=block_mask,
            image_masks=image_masks,
            modality_positions=batch_mod_pos,
            guidance_scale=guidance_scale,
        )

        if guidance_scale > 0:
            block_mask_null = omni_attn_mask_naive(
                B, self.max_seq_len, batch_mod_pos_null, device
            ).to(dtype)
            velocity_fn_kwargs['text_tokens_null'] = batch_text_tokens_null
            velocity_fn_kwargs['block_mask_null'] = block_mask_null
            velocity_fn_kwargs['modality_positions_null'] = batch_mod_pos_null

        return {
            'z_init': z,
            'velocity_fn_kwargs': velocity_fn_kwargs,
        }

    def generate_images_sde(self, prompts: List[str],
                            flow_grpo,
                            num_steps: int = 10,
                            guidance_scale: float = 0.0,
                            ) -> Dict[str, Any]:
        """Generate images using SDE sampling for Flow-GRPO.

        Returns trajectory information needed for GRPO training.

        Args:
            prompts: Text prompts.
            flow_grpo: FlowGRPO instance.
            num_steps: Number of SDE denoising steps.
            guidance_scale: CFG scale (0 = no guidance for training).

        Returns:
            Dict with 'trajectory', 'images', 'latents', 'velocity_fn_kwargs'.
        """
        from utils import denorm

        inputs = self.prepare_flow_grpo_inputs(prompts, guidance_scale)
        z_init = inputs['z_init']
        vfn_kwargs = inputs['velocity_fn_kwargs']

        velocity_fn = self.make_velocity_fn(**vfn_kwargs)

        trajectory = flow_grpo.generate_trajectory(
            velocity_fn, z_init, num_steps=num_steps,
        )

        # Decode final latents
        final_latents = trajectory['final']
        samples_for_vae = final_latents.unsqueeze(2)
        with torch.no_grad():
            images_tensor = self.vae.batch_decode(samples_for_vae).squeeze(2)
            images_np = denorm(images_tensor)
            pil_images = [Image.fromarray(img) for img in images_np]

        return {
            'trajectory': trajectory,
            'images': pil_images,
            'latents': final_latents,
            'velocity_fn_kwargs': vfn_kwargs,
        }

    def compute_flow_grpo_logprobs(
        self,
        flow_grpo,
        trajectory: Dict,
        velocity_fn_kwargs: Dict,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute trajectory log probabilities for Flow-GRPO.

        Args:
            flow_grpo: FlowGRPO instance.
            trajectory: From generate_images_sde().
            velocity_fn_kwargs: kwargs for make_velocity_fn().
            step_indices: Subset of steps for Fast mode.

        Returns:
            [B] total log probability.
        """
        velocity_fn = self.make_velocity_fn(**velocity_fn_kwargs)
        return flow_grpo.compute_trajectory_logprob(
            velocity_fn, trajectory,
            step_indices=step_indices,
        )

    def compute_flow_grpo_kl(
        self,
        flow_grpo,
        ref_velocity_fn,
        trajectory: Dict,
        velocity_fn_kwargs: Dict,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute trajectory KL divergence for Flow-GRPO.

        Args:
            flow_grpo: FlowGRPO instance.
            ref_velocity_fn: Reference model velocity function.
            trajectory: From generate_images_sde().
            velocity_fn_kwargs: kwargs for make_velocity_fn().
            step_indices: Subset of steps for Fast mode.

        Returns:
            [B] total KL divergence.
        """
        velocity_fn = self.make_velocity_fn(**velocity_fn_kwargs)
        return flow_grpo.compute_trajectory_kl(
            velocity_fn, ref_velocity_fn, trajectory,
            step_indices=step_indices,
        )

    # ==================== IMAGE UNDERSTANDING (VOTING) ====================

    def compute_voting_logprobs(self, image_latents: torch.Tensor,
                                question: str,
                                completion_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token log probabilities for GRPO on voting text.

        Teacher-forced forward pass: concatenates the prompt with the
        completion tokens, runs through the LLM, and extracts per-token
        log probabilities for the completion portion.

        Args:
            image_latents: [1, C, H, W] image latents.
            question: Voting prompt text.
            completion_ids: [1, L] token IDs of the completion to score.

        Returns:
            [1, L] per-token log probabilities.
        """
        dtype = self.dtype
        device = self.device

        # Encode image
        image_embeds = self._encode_image_for_mmu(image_latents)

        # Build prompt embeddings
        input_ids = self.tokenizer(question, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor(
            [self.bos_id] + self.sys_prompt_ids + self.role_a_ids
        ).to(device)[None, :]
        text_tokens_b = torch.tensor(
            [self.boi_id, self.eoi_id] + input_ids + self.role_b_ids
        ).to(device)[None, :]

        # Embed all text tokens (prompt + completion)
        all_text_tokens = torch.cat([text_tokens_a, text_tokens_b, completion_ids], dim=1)
        all_text_embeds = self.model.showo.model.embed_tokens(all_text_tokens)

        # Split to insert image embeddings at the correct position
        len_a = text_tokens_a.shape[1]

        if self.add_time_embeds:
            time_embeds = self.model.time_embed(
                torch.Tensor([[1.0]]).to(device), all_text_embeds.dtype
            )
            if hasattr(self.model, 'time_embed_proj'):
                time_embeds = self.model.time_embed_proj(time_embeds)
            # Structure: [text_a | boi | time_embed | image | eoi+question+role_b | completion]
            input_embeds = torch.cat([
                all_text_embeds[:, :len_a],         # text_a embeddings
                all_text_embeds[:, len_a:len_a + 1], # boi embedding
                time_embeds,                          # time embedding
                image_embeds,                         # image embeddings
                all_text_embeds[:, len_a + 1:],       # eoi + question + role_b + completion
            ], dim=1).to(dtype)
            # Number of inserted tokens (time_embed + image - the placeholder space)
            num_image_inserted = 1 + image_embeds.shape[1]  # time + image tokens
        else:
            input_embeds = torch.cat([
                all_text_embeds[:, :len_a],
                all_text_embeds[:, len_a:len_a + 1], # boi
                image_embeds,
                all_text_embeds[:, len_a + 1:],
            ], dim=1).to(dtype)
            num_image_inserted = image_embeds.shape[1]

        # Forward pass through LLM to get logits
        outputs = self.model.showo(
            inputs_embeds=input_embeds,
            output_hidden_states=False,
        )
        logits = outputs['logits']

        # Extract log probs for the completion portion
        # Total prompt length in input_embeds = original text length + inserted image tokens
        completion_len = completion_ids.shape[1]
        # The completion tokens start at: total_len - completion_len
        prompt_total_len = input_embeds.shape[1] - completion_len

        # For next-token prediction: logits at position i predict token at position i+1
        # So to get the log prob of completion_ids[t], we use logits at position (prompt_total_len - 1 + t)
        completion_logits = logits[:, prompt_total_len - 1:-1, :]  # [1, L, vocab]
        log_probs = F.log_softmax(completion_logits, dim=-1)

        # Gather log probs for actual completion tokens
        per_token_logps = log_probs.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)  # [1, L]

        return per_token_logps
