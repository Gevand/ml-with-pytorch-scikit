from transformers import GPT2Model,GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import pipeline, set_seed
import torch
generate = pipeline('text-generation', model='gpt2')
set_seed(123)
results = generate('Hey readers, today is',
                   max_length=20, num_return_sequences=3)
print(results)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentense"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)

model = GPT2Model.from_pretrained('gpt2')
output = model(**encoded_input)
print(output['last_hidden_state'].shape)
print(output[0][0][0].shape, ' -> ', output[0][0][0])

model2 = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
output = model2(**encoded_input)
print(output['logits'].shape)
print(output[0][0][0].shape, ' -> ', output[0][0][0])
print(torch.argmax(output['logits']), ' -> ', output[0][0][0][torch.argmax(output[0][0][0])])
predicted_token_ids = torch.argmax(output['logits'], dim=-1)
output_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(output_text)