# AMREx - Prototype Code and Instructions
The AMREx prototype code submitted with this article is at its core compact in one file `amrex.py`. From a software engineering point of view, it would be questioned why the code is not presented as a repository and extracted into modules. I chose to compact the modules into one executable file because the code should serve as a prototype accompany to this article, where the interested reader can begin to and elaborate on it based on own needs. The code is highly flexible, comprehensive, and compact. This should help the reader to move easily up and down through the one file, or even execute it in a Jupyter Notebook for initial experiments. Furthermore, the user is required to install the packages necessary as indicated on the top of the file, of course.

The prototype `amrex.py` involves logger, and you will need to set the level of logging that you want to go for. In addition to the `amrex.py`, there is a `config.py` file for the overall configuration of the parameters. An `.env` file should also be maintained with the proper IDs and tokens.

### EXAMPLE .env file
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
ASSISTANT_ID_1=asst_...


As for the set-up of the assistant and its instructions, the user will be required to constitute an assistant on the OpenAI dashboard. In our case, the assistant is named "Athena," a name chosen by the assistant itself through auto-development, supported by compelling reasoning. If you want to change the name, you should be attentive to change it also within the instructions. Once, you have created an assistant, apply its ID to your `.env` file and the copy/paste the instructions from the instructions' file `Assistant_Instruction.xml` into the instructions' field of your assistant in the dashboard. For this prototype, we do not use an upload of the instruction file, but this is generally possible if prepared well. The model of choice should be as to date "gpt-4-turbo-preview."

Furthermore, a Streamlit interface module tailored for this prototype is available in the `streamlit_app.py` file. The user is required to make the necessary installation of packages as mentioned at the top of the file, of course. To run the prototype via streamlit the standard command `streamlit run streamlit_app.py` would help which will redirect to the chat interface on the browser.

In summary, the four files, `amrex.py`, `config.py`, `Assistant_Instruction.xml`, and `streamlit_app.py`, are available in the Github repository AMREx. It is important to emphasise explicitly that the aim of the code presented to accompany this article is to provide an educational functional hands-on prototype that would help comprehension of the framework and technique. This means that the code does not account at any point as a product or even semi-product of software.
