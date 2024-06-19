import random
from collections import deque


class ReplayBuffer:
    def __init__(self, train_max_size, valid_max_size, train_to_valid_ratio):
        assert valid_max_size <= train_max_size, \
            "Validation buffer size is bigger than train buffer size"
        self.train_buffer = deque(maxlen=train_max_size)
        self.valid_buffer = deque(maxlen=valid_max_size)
        self.train_ratio = train_to_valid_ratio

    def add(self, batch: []):
        train_batch, valid_batch = self.split_batch(batch)
        self.train_buffer.extend(train_batch)
        self.valid_buffer.extend(valid_batch)

    def split_batch(self, batch):
        batch_size = len(batch)
        split_index = int(self.train_ratio * batch_size)
        random.shuffle(batch)
        train = batch[:split_index]
        valid = batch[split_index:]
        return train, valid

    def feed_valid_into_train_buffer(self):
        self.train_buffer.extend(self.valid_buffer)
        self.valid_buffer.clear()

    def sample(self, buffer_percent):
        batch_size = int(self.train_buffer.maxlen * buffer_percent)
        if batch_size >= len(self.train_buffer):
            train = list(self.train_buffer)
        else:
            train = random.sample(self.train_buffer, batch_size)
        return (train,
                list(self.valid_buffer))



if __name__ == '__main__':
    r = ReplayBuffer(250, 80, 0.66)
    data = list(range(150))
    r.add(data)
    kk = r.sample(0.8)
    r.feed_valid_into_train_buffer()

    data = list(range(250, 390))
    r.add(data)

    data = list(range(150))
    r.add(data)
    r.feed_valid_into_train_buffer()
    kk = r.sample(0.5)

    ss = 0

