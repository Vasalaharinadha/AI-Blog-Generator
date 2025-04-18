import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(
    model='TheBloke/Llama-2-7B-Chat-GGML',
    model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
    model_type='llama',
    config={
        'max_new_tokens': 256,
        'temperature': 0.01
        }
    )

    # Prompt Template
    template = """
    Write a blog for a {blog_style} job profile on the topic "{input_text}".
    The blog should be concise and around {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )

    # Generate the response
    response = llm(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ))

    return response

# Streamlit UI
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.number_input('No of Words', min_value=50, max_value=1000, step=50)
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'))

submit = st.button("Generate")

# Final output
if submit and input_text:
    with st.spinner('Generating...'):
        output = getLLamaresponse(input_text, no_words, blog_style)
        st.subheader("Generated Blog")
        st.write(output)