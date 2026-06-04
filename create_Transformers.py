from transformers import AutoModel
from transformers import AutoTokenizer
model = AutoModel.from_pretrained("bert-base-cased")
# model.save_pretrained("./my_bert_model")

print("Model saved successfully!")

#REUSING THE SAVED MODEL
#model=AutoModel.from_pretrained("./my_bert_model")


tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
print("Encoding :")
encoded_input=tokenizer("HELLO ,I'M SUSHANTH SHETTY","hiiiii")
print(encoded_input)

print("-----------------------------------------------------------------------------------------------------")
print("Decoding the input ids back to text:")
decode_input=tokenizer.decode(encoded_input['input_ids'])
print(decode_input)

# using tensors 
print("------------------------------------------------------------------------------------------------------")
encoded_input=tokenizer("HELLO ,I'M SUSHANTH SHETTY",padding=True, return_tensors='pt')
print(encoded_input)
encoded_input1=tokenizer("hello ,how are you ?",
                         padding=True,
                         truncation=True,
                         max_length=10,
                         return_tensors='pt',
)
print(encoded_input)

