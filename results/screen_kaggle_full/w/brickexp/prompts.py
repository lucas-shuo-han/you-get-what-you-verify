"""Prompt text (copied from BrickGPT's create_instruction_zero_shot so we don't import torch)."""

SYSTEM = "You are a helpful assistant."
GRID = 20
MAX_BRICKS = 300


def instruction(caption: str) -> str:
    return ('Create a LEGO model of the input. Format your response as a list of bricks: '
            '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
            'Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n'
            'All bricks are 1 unit tall.\n\n'
            '### Input:\n'
            f'{caption}'
            '\n\n'
            'Each line of your output should be a LEGO brick in the format `<brick dimensions> <brick position>`. For example:\n'
            '2x4 (2,1,0)\n'
            'DO NOT output any other text. Only output LEGO bricks. The first brick should have a z-coordinate of 0.\n'
            # Added constraints (not in BrickGPT's prompt): grid bounds + size cap, after v6 degenerate long lists
            f'The model must fit in a {GRID}x{GRID}x{GRID} grid: every brick must satisfy 0 <= x, x+H <= {GRID}, '
            f'0 <= y, y+W <= {GRID}, 0 <= z < {GRID}. Use at most {MAX_BRICKS} bricks, one per line, then stop.')


def feedback(check_message: str) -> str:
    return (f'The physics checker rejected your structure. {check_message}\n'
            'Output the complete corrected structure (all bricks, one per line), with no other text.')


def initial_messages(caption: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction(caption)}]
