import random
import numpy as np

# In Hours
SHIFT_LENGTH = 8
BLOCKS_PER_DAY = 24 // SHIFT_LENGTH
DAYS_PEW_WORK_WEEK = 4
BLOCKS_PER_WEEK = BLOCKS_PER_DAY * DAYS_PEW_WORK_WEEK
DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

RNG = np.random.default_rng()


class Shift:
    def __init__(self, start_block, duration, people):
        self.start_block = start_block
        self.duration = duration
        self.people = people

    @property
    def shift_end_block(self):
        return self.start_block + self.duration

    @staticmethod
    def real_time(start_of_time=0, delta_blocks=0):

        block_time = start_of_time + delta_blocks
        td_ratio = (block_time/BLOCKS_PER_DAY)
        real_t = td_ratio % 24
        real_d = int(td_ratio // 24)

        print(f'Real Day Real Time of {real_t}')
        return DAYS[real_d % 7], real_t


def get_people():
    return ['A', 'B', 'C', 'D']


def gen_random_schedule(people, amount=1):
    return RNG.integers(low=0, high=len(people), size=(amount, BLOCKS_PER_WEEK), dtype=np.int8)


def eval_schedule(schedules, amount_people):
    def _roll_mask(maskk, steps, shift_amount=1):
        for i in range(maskk.shape[-1]-(1+steps)):
            maskk[:, :, i] = np.roll(maskk[:, :, 0], shift=-i*shift_amount, axis=-1)
        return maskk

    def block_distrobution(block_mask):
        #add check per day
        sum_of_blocks = block_mask.sum(axis=-1)
        div_blocks = sum_of_blocks.std(axis=-1)
        return div_blocks * -10

    def sleep_check(block_mask):
        sleep_blocks = BLOCKS_PER_DAY // 3
        max_shift_blocks = BLOCKS_PER_DAY // 2

        shape = block_mask.shape
        b = block_mask.reshape(shape[0], shape[1], 1, shape[2])
        e = np.ones(shape=(1, 1, shape[-1], 1), dtype=bool)
        unrolled_mask = b & e
        rolling_non_sleep_days = _roll_mask(unrolled_mask, sleep_blocks)[:, :, :-(BLOCKS_PER_DAY-1), :BLOCKS_PER_DAY]
        #print(rolling_non_sleep_days)

        rolling_pair_mask = _roll_mask(unrolled_mask, 1)[:, :, :-1, :2]

        starting_shifts = (rolling_pair_mask == [False, True]).all(axis=-1)
        #index_start = np.argwhere(starting_shifts)
        ending_shifts = (rolling_pair_mask == [True, False]).all(axis=-1)
        #index_end = np.argwhere(ending_shifts)

        print(f'start_of_shift{starting_shifts[0, 0]}\r\n')
        print(f'end_of_shift{ending_shifts[0, 0]}')

        #print(f'start:{index_start}')
        #print(starting_shifts[index_start])
        #print(f'end:{index_end}')
        #print(starting_shifts[index_end])

        #total_shift_count = ending_shifts.sum(axis=-1)



        #weight = (total_shift_count.sum(axis=-1) * -100) #(has_days_sleep.sum(axis=-1) * -200) + (total_shift_count.sum(axis=-1) * -100)  # +(has_sleep.sum(axis=-1) * -100)
        #return weight

    def shifts_per_day_check(block_mask):
        shape = block_mask.shape
        b = block_mask.reshape(shape[0], shape[1], 1, shape[2])
        e = np.ones(shape=(1, 1, shape[-1], 1), dtype=bool)
        sliced_rolled_days = _roll_mask(b & e, BLOCKS_PER_DAY, BLOCKS_PER_DAY)[:, :, :, :BLOCKS_PER_DAY]
        shifts_per_day = sliced_rolled_days.sum(axis=-1)[:, :, :DAYS_PEW_WORK_WEEK]
        weights = (shifts_per_day > 1).sum(axis=-1).sum(axis=-1) * -10
        return weights


    people_indices = np.arange(0, amount_people)
    p = people_indices.reshape((1, -1, 1))
    s = schedules.reshape((schedules.shape[0], 1, schedules.shape[1]))
    mask = np.equal(s, p)
    print(mask.shape)

    #weight_table = sleep_weight
    print(f'Schedule[0]: {schedules[0]}\r\n')

    sleep_penalty = sleep_check(mask)

    multi_shift_penalty = shifts_per_day_check(mask)
    print(f'multi_shift_penalty :{multi_shift_penalty.max()}\r\n')

    block_distrobution_weight = block_distrobution(mask)
    print(f'block_distrobution_weight :{block_distrobution_weight.max()}\r\n')

    weight_table = block_distrobution_weight + multi_shift_penalty
    return schedules[weight_table.argmax()]




def make_schedule():
    people = get_people()
    return eval_schedule(gen_random_schedule(people, 10000), len(people))


if __name__ == "__main__":
    print(f'Result: {make_schedule()}')