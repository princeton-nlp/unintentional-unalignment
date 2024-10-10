import copy

import numpy as np
from torch.utils.data.sampler import Sampler


class SameClassBatchSampler(Sampler):

    def __init__(self, labels, batch_size, num_same_class_in_batch):
        self.labels = labels
        self.by_labels_indices = self.__create_by_labels_indices(labels)
        self.batch_size = batch_size
        self.num_same_class_in_batch = num_same_class_in_batch

    def __create_by_labels_indices(self, labels):
        by_labels_indices = {}
        for i in range(len(labels)):
            label = labels[i]
            if label not in by_labels_indices:
                by_labels_indices[label] = []

            by_labels_indices[label].append(i)

        return by_labels_indices

    def __iter__(self):
        curr_by_labels_indices = copy.deepcopy(self.by_labels_indices)
        while curr_by_labels_indices:
            batch_indices = []
            while len(batch_indices) < self.batch_size and curr_by_labels_indices:
                optional_labels = list(curr_by_labels_indices.keys())
                chosen_label = optional_labels[np.random.randint(len(optional_labels))]
                label_sample_indices = curr_by_labels_indices[chosen_label]

                num_in_class_to_sample = min(self.num_same_class_in_batch, self.batch_size - len(batch_indices), len(label_sample_indices))
                sampled_indices = np.random.choice(label_sample_indices, num_in_class_to_sample, replace=False).tolist()

                curr_by_labels_indices[chosen_label] = [i for i in label_sample_indices if i not in sampled_indices]
                if not curr_by_labels_indices[chosen_label]:
                    del curr_by_labels_indices[chosen_label]

                batch_indices.extend(sampled_indices)

            yield batch_indices

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size
