from __future__ import annotations  # Allows for using the class name in type hints within the class definition, 
                                    # and it has to be the first line of the file
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True) # Define an immutable dataclass for storing the state of each step
class ScheduleStep:
    t: int
    level: int
    is_transition: bool
    prev_level: Optional[int] = None    # For logging/helper purposes


from typing import Sequence, List, Iterator

class MultiScaleSchedule:
    def __init__(self, level_of_t: Sequence[int]):
        self._level_of_t: List[int] = [int(x) for x in level_of_t]
    
    @property
    def T(self) -> int:
        return len(self._level_of_t)
    
    # The main method to iterate through the schedule in reverse order, yielding a ScheduleStep for each time step
    def iter_reverse(self, t_start: Optional[int] = None) -> Iterator[ScheduleStep]:
        if t_start is None:
            t_start = self.T - 1

        for t in range(t_start, -1, -1):
            level = self._level_of_t[t]

            prev_level = self._level_of_t[t - 1] if t > 0 else level  # For t=0, we can consider prev_level the same as current level

            # A transition happens when we change level compared to the previous step
            is_transition = (t > 0) and (level != prev_level)

            # yield the current step with the transition information
            yield ScheduleStep(t=t, level=level, is_transition=is_transition, prev_level=prev_level)

    # For convenience, we use transition time steps to create the level_of_t list instead of by hands
    @classmethod    # For user convenience
    def from_block_spec(
        cls, 
        T: int, 
        levels: Sequence[int], 
        transition_ts: Sequence[int]
    ) -> MultiScaleSchedule:
        # Defensive checks for input validity
        if T <= 0:
            raise ValueError(f"T must be a positive integer, got {T}.")
        if len(levels) != len(transition_ts) + 1:
            raise ValueError(
                f"Need len(levels) == len(transition_ts) + 1, got {len(levels)} vs {len(transition_ts)}."
            )
        
        transition_ts = [int(x) for x in transition_ts] # Ensure/Transform transition_ts are/to integers
        levels = [int(x) for x in levels] # Ensure/Transform levels are/to integers

        for t in transition_ts:
            if not (0 <= t <= T - 1):
                raise ValueError(f"Transition times steps out of range: {t} not in [0, {T-1}].")
        if any(transition_ts[i] <= transition_ts[i + 1] for i in range(len(transition_ts) - 1)):
            raise ValueError("transition_ts must be in strictly decreasing order, e.g. [700,200].")

        # Core logics: Fill in the level_of_t list based on the transition points and levels
        level_of_t = [levels[0]] * T  # Start with all steps at the first level
        boundaries = [T - 1] + transition_ts + [-1]  # Add T-1 at the start and -1 at the end for easier iteration

        for idx, level in enumerate(levels):
            high = boundaries[idx]
            low = boundaries[idx + 1]
            # Fill in the level for the steps in the current block
            for t in range(high, low, -1):
                level_of_t[t] = level

        return cls(level_of_t)  # Same as MultiScaleSchedule(level_of_t), which is a instance of the class


# The manual list was: [2, 2, 1, 1, 0, 0]
# Total elements (T) = 6
# Levels used = [2, 1, 0]
# Transition points (where the level changes when moving 5 -> 0):
# t=3 is where it changes to 1
# t=1 is where it changes to 0

# smoke test (print the schedule steps in reverse order to verify correctness)
def _demo():
    schedule = MultiScaleSchedule.from_block_spec(T=6, levels=[2, 1, 0], transition_ts=[3, 1])
    for step in schedule.iter_reverse():
        print(step)

# assert test to verify the correctness of the schedule generation and iteration logic
def _self_test():
    schedule = MultiScaleSchedule.from_block_spec(T=6, levels=[2, 1, 0], transition_ts=[3, 1])

    # Test with default starting point (t_start=None, which should start from T-1=5)
    steps = list(schedule.iter_reverse())

    assert [s.t for s in steps] == [5, 4, 3, 2, 1, 0]   # Using property t from ScheduleStep to access the time steps
                                                        # The time steps should be in reverse order from T-1 down to 0
    assert [s.level for s in steps] == [2, 2, 1, 1, 0, 0]   # The levels should match the expected pattern based on the transition points
    assert [s.is_transition for s in steps] == [False, True, False, True, False, False] # The is_transition should be True at the points 
                                                                                        # where the level changes (t=3 and t=1) and False otherwise

    # Test transition semantics used by the sampler.
    # In the sampler, each reverse update is x_i -> x_{i-1}.
    # Therefore, a scale transition happens at loop index i when:
    #   level(i) != level(i-1)
    #
    # For:
    #   t:      5 4 3 2 1 0
    #   level:  2 2 1 1 0 0
    #
    # The actual reverse-update transitions are:
    #   i=4: x_4(level 2) -> x_3(level 1)
    #   i=2: x_2(level 1) -> x_1(level 0)
    sampler_transition_indices = []
    sampler_transition_flags = []

    for i in range(schedule.T - 1, -1, -1):
        level_t = schedule._level_of_t[i]
        level_t_prev = schedule._level_of_t[i - 1] if i > 0 else level_t
        is_transition = (i > 0) and (level_t != level_t_prev)

        sampler_transition_flags.append(is_transition)
        if is_transition:
            sampler_transition_indices.append(i)

    assert sampler_transition_flags == [False, True, False, True, False, False]
    assert sampler_transition_indices == [4, 2]

    # Test with a different starting point (t_start=3)
    steps2 = list(schedule.iter_reverse(t_start=3))
    assert [s.t for s in steps2] == [3, 2, 1, 0]
    assert [s.level for s in steps2] == [1, 1, 0, 0]
    assert [s.is_transition for s in steps2] == [False, True, False, False]

    # Invalid transition order should raise
    try:    # If a 'wrong' schedule is created with non-decreasing transition_ts successfully, 
            # then we raise an AssertionError to indicate the test failure. 
            # We expect a ValueError to be raised due to the invalid input.
        MultiScaleSchedule.from_block_spec(T=6, levels=[2, 1, 0], transition_ts=[1, 3])
        raise AssertionError("Expected ValueError for non-decreasing transition_ts.")
    except ValueError:  # If the 'wrong' schedule creation raises a ValueError as expected, we pass the test.
        pass

if __name__ == "__main__":
    _self_test()
    _demo()
