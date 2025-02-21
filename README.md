Vector-DB-Class
===============

This repository demonstrates various Python scripts, Jupyter notebooks, and a Streamlit application for working with gene data and vector databases.

Table of Contents
-----------------

1.  [Repository Contents](#repository-contents)
    
2.  [Cloning the Repository](#cloning-the-repository)
    
3.  [Setting Up the Environment](#setting-up-the-environment)
    
4.  [Environment Variables](#environment-variables)
    
5.  [Running Python Scripts](#running-python-scripts)
    
6.  [Running Streamlit Apps](#running-streamlit-apps)
    
7.  [Running Jupyter Notebooks](#running-jupyter-notebooks)
    
8.  [Data Files and Other Resources](#data-files-and-other-resources)
    
9.  [License](#license)
    

Repository Contents
-------------------

Within the repository, you will find:

*   **.env** (environment variables file)
    
*   **cancer\_gene\_data\_100.csv** (sample CSV data)
    
*   **fundamentals.ipynb** (Jupyter notebook)
    
*   **fudamentals.ipynb** (typo? Possibly a duplicate or alternative notebook)
    
*   **Gene-expression-example.py** (Python script)
    
*   **intro.ipynb** (Jupyter notebook)
    
*   **RAG-example.py** (Streamlit application)
    
*   **RAG-GENE.py** (Streamlit application)
    
*   **requirements.txt** (Python dependencies)
    
*   **synthetic\_gene\_ecpression.py** (Python script)
    
*   **Vectordatabase.pptx** (PowerPoint presentation)
    

Cloning the Repository
----------------------

To get started, clone the repository from GitHub (replace the URL with your actual repo URL):

`   [git clone https://github.com/yourusername/Vector-DB-Class.git](https://github.com/rahulsharma-rs/Vector-DB-Class.git)   `

Then move into the project directory:

`   cd Vector-DB-Class   `

Setting Up the Environment
--------------------------

It is recommended to use a virtual environment to keep your dependencies organized.

1.  `python -m venv venv`
    
2.  **Activate the virtual environment**:
    
    *   `source venv/bin/activate`
        
        
3.  `pip install -r requirements.txt`
    

Environment Variables
---------------------

This project uses a .env file to store environment variables such as your GPT key.

1.  Create a .env file in the root folder if it does not exist already.
    
2.  `GPT_KEY=your_gpt_key_here`
    

Replace your\_gpt\_key\_here with your actual GPT key (e.g., from OpenAI).

Running Python Scripts
----------------------

There are several standalone Python scripts in this repository. To run any of them, simply use:

`   python .py   `

Examples:

*   `python Gene-expression-example.py`
    
*   `python synthetic_gene_ecpression.py`
    

Running Streamlit Apps
----------------------

There are two Streamlit applications in this repo: **RAG-example.py** and **RAG-GENE.py**.

To launch either app, run:

`   streamlit run RAG-example.py   `

or

`   streamlit run RAG-GENE.py   `

This will start a local server and open the app in your web browser.

Running Jupyter Notebooks
-------------------------

For any Jupyter Notebook (e.g., fundamentals.ipynb, intro.ipynb):

1.  `jupyter notebook`
    
2.  In the browser window that opens, navigate to the notebook you want to open and click on it to launch.
    

Data Files and Other Resources
------------------------------

*   **cancer\_gene\_data\_100.csv**: Sample CSV file containing gene data.
    
*   **Vectordatabase.pptx**: A PowerPoint presentation that may provide additional background or context for the project.
    

License
-------

Specify your project’s license here (e.g., MIT, Apache 2.0, etc.) or remove this section if not applicable.

**Enjoy exploring and using the Vector-DB-Class project!** If you encounter any issues or have suggestions, feel free to open an issue or submit a pull request.
