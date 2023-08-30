# importing all the necessary files

from IPython.display import YouTubeVideo

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
import locale
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch

import langchain
print(langchain.__version__)

#Loading a sample video into transcript

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=tAuRQs_d9F8&t=52s")
transcript = loader.load()

# Recursive splitting of text and storing it into texts

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
texts = text_splitter.split_documents(transcript)

# Loading the model

model_repo = 'tiiuae/falcon-rw-1b'

tokenizer = AutoTokenizer.from_pretrained(model_repo)

model = AutoModelForCausalLM.from_pretrained(model_repo,
                                             load_in_8bit=True,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True,
                                             trust_remote_code=True
                                            )
max_len = 2048 # 1024
task = "text-generation"
T = 0

# Building the pipeline

pipe = pipeline(
    task=task,
    model=model, 
    tokenizer=tokenizer, 
    max_length=max_len,
    temperature=T,
    top_p=0.95,
    repetition_penalty=1.15,
    pad_token_id = 11
)

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs = {'temperature':0})

#Intitializing the LLM chain

template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

locale.getpreferredencoding = lambda: "UTF-8"

# import and intialize the question answer pipeline

model_checkpoint = "IProject-10/bert-base-uncased-finetuned-squad2"
question_answerer = pipeline("question-answering", model=model_checkpoint)

text1 = """{}""".format(transcript[0])[14:]

context = text1

# Get the context of the video

def get_context(input_text):
   loader = YoutubeLoader.from_youtube_url("{}".format(input_text))
   transcript = loader.load()
   texts = text_splitter.split_documents(transcript)
   text1 = """{}""".format(transcript[0])[14:]
   context = text1
   return context

# Building the bot function

def build_the_bot(text1):
  context = text1
  return('Bot Build Successfull!!!')

# Building the bot summarizer function

def build_the_bot_summarizer(text1):
  text = text1
  return llm_chain.run(text)

# The chat space for gradio is servered here

def chat(chat_history, user_input, context):

  output = question_answerer(question=user_input, context=context)
  bot_response = output["answer"]
  #print(bot_response)
  response = ""
  for letter in ''.join(bot_response): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
      response += letter + ""
      yield chat_history + [(user_input, response)]

# Serving the entre gradio app

with gr.Blocks() as demo:
    gr.Markdown('# YouTube Q&A and Summarizer Bot')
    with gr.Tab("Input URL of video you wanna load -"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button1 = gr.Button("Build the Bot!!!")
        text_button1.click(build_the_bot, get_context(text_input), text_output)
        text_button2 = gr.Button("Summarize...")
        text_button2.click(build_the_bot_summarizer, get_context(text_input), text_output)
    with gr.Tab("Knowledge Base -"):
#          inputbox = gr.Textbox("Input your text to build a Q&A Bot here.....")
          chatbot = gr.Chatbot()
          message = gr.Textbox ("What is this Youtube Video about?")
          message.submit(chat, [chatbot, message], chatbot, get_context(text_input))

demo.queue().launch()
