# Urban Greening Planner - An Interactive Dashboard for Tree Planting in Vancouver

- Student Name: Lakshantha Dissanayake
- Student ID: 300392299
- Email ID: dissanayakel@student.douglascollege.ca

# Steps to run the web application

## Prerequisites

- Make sure [Python3](https://www.python.org/downloads) is installed on the system.

## Clone the app


```sh
git clone https://github.com/emergingTechExplorer/F2025_4495_071_LDi299
cd F2025_4495_071_LDi299/Implementation
```

## Create a Python virtual environment


```sh
python3 -m venv env 
source env/bin/activate
```

## Install necessary packages

```sh
pip install -r requirements.txt
```

## Create a Streamlit secrets file

```sh
mkdir .streamlit
touch .streamlit/secrets.toml
```

## Add the following entries inside the secrets file

```sh
OPENAI_API_KEY = “<your_openai_api_key>”
OPENAI_MODEL = “gpt-5-mini”
```

If this configuration is omitted, all the functionalities of the application remain intact, but AI summaries are disabled.

## Run the app

```sh
streamlit run app.py
```

The application will open automatically in your default browser at http://localhost:8501