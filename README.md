# Sustainable Data Day - ESG Data Bot Retriever

🤖Retrieve ESG Data from corporate report in natural language🤖

**Note**: this project is adapted from [Harrison Chase](https://github.com/hwchase17/) [Streamlit LLM chat bot](https://github.com/hwchase17/notion-qa) to handle PDF and Fanilo Andrianasolo [modifications](https://www.youtube.com/watch?v=yZmOxIBiWQI) for caching and improve user interactions.


💪 Built with [LangChain](https://github.com/hwchase17/langchain)

# 🌲 Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

For streamlit, add your key in a OPENAI_API_KEY entry in your secrets.toml

# 📄 What is in here?
- Example data from french corporates. 
- Python script to query pdf with a question
- Code to deploy on StreamLit
- Instructions for ingesting your own dataset

## 📊 Example Data
This repo uses the [Total Energies](https://totalenergies.com/sites/g/files/nytnzq121/files/documents/2023-03/TotalEnergies_DEU_2022_VF.pdf) corporate report as an example.
It was downloaded August 2023.

## 💬 Ask a question
In order to ask a question, run a command like:

```shell
python cli.py
```

You can then asked any question of your liking!

This exposes a chat interface for interacting with the PDF database.

## 🚀 Code to deploy on StreamLit

The code to run the StreamLit app is in `main.py`. 
Note that when setting up your StreamLit app you should make sure to add `OPENAI_API_KEY` as a secret environment variable.

## 🧑 Instructions for ingesting your own dataset

You can add any corporate reports of your liking in the PDF directory. Launch ingest.py to add them in the local vector store.

Run the following command to ingest the data.

```shell
python ingest.py
```

Boom! Now you're done.

We provide a vector database initialized on Total Energies and Engie 2022 corporate reports and their wikipedia page.