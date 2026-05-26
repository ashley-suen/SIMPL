import sys
sys.path.append('e:/Msc/new/code/simpl_new/SIMPL/data_av2/')
from av2_llm_dataset import AV2PromptDataset

dataset = AV2PromptDataset('e:/Msc/new/code/simpl_new/SIMPL/data_av2/features/train/')
# Let's inspect the next 5 samples
for i in range(1, 6):
    print(f'\n\n=== Sample {i} ===')
    prompt = dataset[i]['prompt_text']
    
    # Just print the neighbor section to save output space
    lines = prompt.split('\n')
    in_neighbor = False
    for line in lines:
        if line.startswith('[NEIGHBOR AGENTS HISTORY]'):
            in_neighbor = True
        elif line.startswith('[LOCAL MAP TOPOLOGY & BOUNDARIES]'):
            in_neighbor = False
        
        if in_neighbor:
            print(line)
