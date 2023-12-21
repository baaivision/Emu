"""Reader is module to read the url list and return shards"""
import braceexpand


class OutputSharder:
    """
    The reader class reads a shard list and returns shards

    It provides an iter method
    It provides attributes:
    - shard_list: a list of shards to read
    - input_format: the format of the input dataset
    - done_shards: a set of already done shards
    - group_shards: the number of shards to group together
    """

    def __init__(
        self,
        shard_list,
        input_format,
        done_shards,
    ) -> None:

        self.input_format = input_format
        self.done_shards = done_shards
        self.shard_list = list(braceexpand.braceexpand(shard_list))

        if self.input_format == "webdataset":
            self.shard_ids = [s.split("/")[-1][: -len(".tar")] for s in self.shard_list]
        elif self.input_format == "files":
            self.shard_ids = [s.split("/")[-1] for s in self.shard_list]

        self.shards = [
            (s, s_id) for s_id, s in zip(self.shard_ids, self.shard_list) if int(s_id) not in self.done_shards
        ]

    def __iter__(self):
        """
        Iterate over shards, yield shards of size group_shards size
        Each shard is a tuple (shard_id, shard)
        """
        for s, s_id in self.shards:
            yield (s, s_id)
