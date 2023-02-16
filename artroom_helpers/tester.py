from prompt_parsing import weights_handling, split_weighted_subprompts
prompt = "(test) [other test] empty followed by new"

output = weights_handling(prompt)
print(output)

subprompts, weights = split_weighted_subprompts(output)
print(subprompts)
print(weights)