from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

####
# Text generation:
input_string = f"{'You:':<10} >> "
print("Starting Conversation. Type \"EXIT\" to stop or \"RESTART\" to start again.")

starting = True
user_text = input(input_string)

while user_text != "EXIT":
    # encode the input and add end of string token
    input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
    # concatenate new user input with chat history (if there is)
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if not starting else input_ids
    # generate a bot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    #print the output
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"{'DialoGPT:':<10} >> {output}")

    user_text = input(input_string)

    if user_text == "RESTART":
        starting = True
        print('Restarting...')
        user_text = input(input_string)
    else:
        starting = False