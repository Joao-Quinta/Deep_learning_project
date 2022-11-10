import torch
import  dlc_practical_prologue as prologue

def get_sets(nb):
    sets = prologue.generate_pair_sets(nb)
    
    train_set = sets[0]
    train_target = sets[1]
    train_classes = sets[2]

    test_set = sets[3]
    test_target = sets[4]
    test_classes = sets[5]

    return train_set, train_target, train_classes, test_set, test_target, test_classes


if __name__ == "__main__":
    sample_size = 1000
    train_set, train_target, train_classes, test_set, test_target, test_classes = get_sets(sample_size)