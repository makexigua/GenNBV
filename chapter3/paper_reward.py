"""Explicit collision-only terminal penalty; magnitude is NOT specified by paper."""
def collision_terminal_reward(collision, weight):
    if weight <= 0:
        raise ValueError('collision penalty magnitude must be positive')
    return -collision.float() * weight
