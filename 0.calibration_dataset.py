from datasets import load_dataset

if __main__=="__name__":
    dolphin_r1 = load_dataset(
        'cognitivecomputations/dolphin-r1',
        "nonreasoning",
        cache_dir="../dolphin-r1"
    )
    
    dolphin_r1 = load_dataset(
        'cognitivecomputations/dolphin-r1',
        "reasoning-deepseek",
        cache_dir="../dolphin-r1"
    )
    
    dolphin_r1 = load_dataset(
        'cognitivecomputations/dolphin-r1',
        "reasoning-flash",
        cache_dir="../dolphin-r1"
    )