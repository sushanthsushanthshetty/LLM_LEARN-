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

Token ID   Token                Meaning
============================================================
101        [CLS]                Special: Start of sequence
145        HELLO                Word 1
21678      ,                    Punctuation
2162       I                    Word 2
2346       '                    Apostrophe (separate!)
117        M                    Letter M (part of I'M)
146        SUSAN                Word 3 (part of SUSHANTH)
112        ##THAN               Subword (## means continuation)
150        ##-                  Dash (separate)
156        ##TH                 Subword continuation
13329      SHETTY               Word 4
11612      ?                    Question mark
...
102        [SEP]                Special: End of sequence