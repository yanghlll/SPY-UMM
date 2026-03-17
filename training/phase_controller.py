"""
Phase Controller for Interactive Training.

Controls which training phase is active, supporting interactive alternation
between image generation and voting phases.

Adapted from Vision-Zero's interactive phase switching logic
(grpo_trainer.py lines 910-931).
"""


class PhaseController:
    """Controls the active training phase.

    Modes:
    - 'generation': Only train image generation phase
    - 'voting': Only train voting phase
    - 'both': Train both phases every step
    - 'interactive': Alternate between phases every cycle_length steps
    """

    def __init__(self, mode: str = 'interactive', cycle_length: int = 10):
        self.mode = mode
        self.cycle_length = cycle_length

    def get_active_phase(self, global_step: int) -> str:
        """Get the active training phase for the current step."""
        if self.mode == 'interactive':
            total_cycle = self.cycle_length * 2
            position = global_step % total_cycle
            if position < self.cycle_length:
                return 'generation'
            else:
                return 'voting'
        return self.mode

    def should_train_generation(self, global_step: int) -> bool:
        """Whether to train the generation phase at this step."""
        phase = self.get_active_phase(global_step)
        return phase in ('generation', 'both')

    def should_train_voting(self, global_step: int) -> bool:
        """Whether to train the voting phase at this step."""
        phase = self.get_active_phase(global_step)
        return phase in ('voting', 'both')

    def log_phase_info(self, global_step: int) -> str:
        """Get a log-friendly description of current phase."""
        phase = self.get_active_phase(global_step)
        if self.mode == 'interactive':
            total_cycle = self.cycle_length * 2
            position = global_step % total_cycle
            return (
                f"[INTERACTIVE] Step {global_step}, "
                f"Cycle pos {position}/{total_cycle}, "
                f"Active: {phase.upper()}"
            )
        return f"[{self.mode.upper()}] Step {global_step}"
