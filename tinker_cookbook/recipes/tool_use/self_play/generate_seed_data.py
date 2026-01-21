from datasets import load_dataset


if __name__ == "__main__":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2025-26", split="train")
    fw = fw.train_test_split(test_size=0.1, seed=42)
    # save the train and test datasets to jsonl files
    # for the train set, we will randomly sample 10000 documents
    
    train_ds = fw['train'].shuffle(seed=42).select(range(10000))
    test_ds = fw['test'].shuffle(seed=42).select(range(1000))

    train_ds.to_json("train.jsonl", orient="records")
    test_ds.to_json("test.jsonl", orient="records")

