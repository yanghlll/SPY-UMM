"""
Scene Description Generator for SPY-UMM.

Procedurally generates CLEVR-style scene descriptions as (original, modified) pairs.
The modified version replaces 2 objects with different shape/color, keeping positions.
"""

import random
from typing import Tuple, Dict, List, Any


# CLEVR-style object properties
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'brown', 'gray']
SHAPES = ['cube', 'sphere', 'cylinder']
SIZES = ['small', 'large']
MATERIALS = ['metallic', 'rubber']

# Spatial relations for describing object positions
POSITIONS = [
    'to the left of', 'to the right of', 'in front of', 'behind',
    'above', 'below', 'next to'
]

ABSOLUTE_POSITIONS = [
    'on the left side', 'on the right side', 'in the center',
    'in the front', 'in the back', 'in the far left corner',
    'in the far right corner', 'near the center'
]


class SceneDescriptionGenerator:
    """Generates paired scene descriptions for the spy game.

    For each game, generates:
    - original_description: Natural language description of a CLEVR-like scene
    - modified_description: Same scene but with 2 objects replaced (different shape/color)
    """

    def __init__(self, num_objects_min: int = 3, num_objects_max: int = 6,
                 num_to_modify: int = 2):
        self.num_objects_min = num_objects_min
        self.num_objects_max = num_objects_max
        self.num_to_modify = num_to_modify

    def _generate_object(self, rng: random.Random,
                         exclude: List[Tuple[str, str, str, str]] = None) -> Dict[str, str]:
        """Generate a random object with properties, avoiding duplicates."""
        exclude = exclude or []
        for _ in range(100):
            obj = {
                'color': rng.choice(COLORS),
                'shape': rng.choice(SHAPES),
                'size': rng.choice(SIZES),
                'material': rng.choice(MATERIALS),
            }
            key = (obj['color'], obj['shape'], obj['size'], obj['material'])
            if key not in exclude:
                return obj
        # Fallback: return anyway
        return obj

    def _generate_scene(self, rng: random.Random) -> List[Dict[str, str]]:
        """Generate a random scene with N objects."""
        num_objects = rng.randint(self.num_objects_min, self.num_objects_max)
        objects = []
        used_keys = []

        for i in range(num_objects):
            obj = self._generate_object(rng, exclude=used_keys)
            obj['position'] = rng.choice(ABSOLUTE_POSITIONS)
            obj['index'] = i
            objects.append(obj)
            used_keys.append((obj['color'], obj['shape'], obj['size'], obj['material']))

        return objects

    def _modify_scene(self, rng: random.Random,
                      objects: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[int]]:
        """Create a modified version of the scene by replacing some objects."""
        modified = [dict(obj) for obj in objects]  # Deep copy

        # Choose which objects to modify
        num_modify = min(self.num_to_modify, len(objects))
        modify_indices = rng.sample(range(len(objects)), num_modify)

        existing_keys = [
            (o['color'], o['shape'], o['size'], o['material']) for o in objects
        ]

        for idx in modify_indices:
            old_obj = modified[idx]
            # Generate a different object at the same position
            new_obj = self._generate_object(rng, exclude=existing_keys)
            new_obj['position'] = old_obj['position']  # Keep same position
            new_obj['index'] = old_obj['index']
            modified[idx] = new_obj
            # Update existing keys
            existing_keys[idx] = (new_obj['color'], new_obj['shape'],
                                  new_obj['size'], new_obj['material'])

        return modified, modify_indices

    def _describe_object(self, obj: Dict[str, str]) -> str:
        """Generate a natural language description of a single object."""
        return f"a {obj['size']} {obj['color']} {obj['material']} {obj['shape']}"

    def _describe_scene(self, objects: List[Dict[str, str]],
                        style: str = 'list') -> str:
        """Generate a natural language description of the entire scene.

        Args:
            objects: List of object dicts.
            style: Description style ('list', 'narrative', 'structured').
                   Must be the same for original and modified to avoid
                   format leaking the spy's identity.
        """
        if not objects:
            return "An empty scene."

        descriptions = []

        # Describe each object with its position
        for i, obj in enumerate(objects):
            obj_desc = self._describe_object(obj)
            pos_desc = obj['position']
            descriptions.append(f"{obj_desc} {pos_desc}")

        # Build the full description
        if len(descriptions) == 1:
            return f"A scene with {descriptions[0]}."

        if style == 'list':
            items = ', '.join(descriptions[:-1]) + f', and {descriptions[-1]}'
            return f"A scene containing {items}."

        elif style == 'narrative':
            parts = []
            for i, desc in enumerate(descriptions):
                if i == 0:
                    parts.append(f"There is {desc}")
                elif i == len(descriptions) - 1:
                    parts.append(f"and {desc}")
                else:
                    parts.append(desc)
            return '. '.join(parts) + '.'

        else:  # structured
            lines = [f"Scene description:"]
            for i, desc in enumerate(descriptions):
                lines.append(f"- Object {i+1}: {desc}")
            return ' '.join(lines)

    def generate_pair(self, seed: int) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a (original_description, modified_description, metadata) pair.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (original_desc, modified_desc, metadata)
        """
        rng = random.Random(seed)

        # Generate original scene
        objects = self._generate_scene(rng)

        # Create modified version
        modified_objects, modify_indices = self._modify_scene(rng, objects)

        # Pick style once, use same style for both (prevents format leak)
        style = rng.choice(['list', 'narrative', 'structured'])
        original_desc = self._describe_scene(objects, style=style)
        modified_desc = self._describe_scene(modified_objects, style=style)

        # Build metadata about differences
        differences = []
        for idx in modify_indices:
            differences.append({
                'position_index': idx,
                'original': {
                    'color': objects[idx]['color'],
                    'shape': objects[idx]['shape'],
                    'size': objects[idx]['size'],
                    'material': objects[idx]['material'],
                },
                'modified': {
                    'color': modified_objects[idx]['color'],
                    'shape': modified_objects[idx]['shape'],
                    'size': modified_objects[idx]['size'],
                    'material': modified_objects[idx]['material'],
                }
            })

        metadata = {
            'num_objects': len(objects),
            'num_modified': len(modify_indices),
            'modify_indices': modify_indices,
            'differences': differences,
            'original_objects': objects,
            'modified_objects': modified_objects,
        }

        return original_desc, modified_desc, metadata


if __name__ == '__main__':
    gen = SceneDescriptionGenerator()
    for i in range(5):
        orig, mod, meta = gen.generate_pair(seed=i * 100)
        print(f"=== Game {i+1} ===")
        print(f"Original:  {orig}")
        print(f"Modified:  {mod}")
        print(f"Changes:   {len(meta['differences'])} objects modified")
        for diff in meta['differences']:
            print(f"  - Position {diff['position_index']}: "
                  f"{diff['original']['color']} {diff['original']['shape']} -> "
                  f"{diff['modified']['color']} {diff['modified']['shape']}")
        print()
