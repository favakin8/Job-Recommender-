# JobRecommenderProject

A command-line career recommendation tool that matches users to jobs based on their skills, qualifications, and salary expectations using Natural Language Processing (NLP) and machine learning.

IMPORTANT NOTES:
You will have to download the data source: [(https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)]
License: CC0: Public Domain, (https://creativecommons.org/publicdomain/zero/1.0/)


So what's it about?:
"Feeling stuck on your career path? This tool takes a short questionnaire about your skills, preferred field, and qualifications, then recommends the most suitable jobs from a dataset of thousands of listings — ranked by relevance and salary."

- The user answers a short questionnaire (field, skills, qualifications, minimum salary)
- Their responses are cleaned, lemmatized, and expanded using a synonym dictionary
- A TF-IDF vectorizer converts job listings and the user profile into numerical vectors
- Cosine similarity measures how closely each job matches the user profile
- A final score is calculated (70% similarity + 30% salary score)
- The top matching jobs are displayed with titles, salaries, descriptions, and match scores

Technologies Used:
- Python!!
- pandas — data loading and manipulation
- scikit-learn — TF-IDF vectorization and cosine similarity
- NLTK — text lemmatization
- pickle — caching processed data for faster future runs

How to use:

1. Clone the Repository
bashgit clone https://github.com/favakin8/JobRecommenderProject.git
cd JobRecommenderProject

2. Install Dependencies
bashpip install -r requirements.txt

3. Add the Dataset
Download the job descriptions dataset and place it in the project folder as:
'job_descriptions.csv'

4. Run the Program
bashpython job-recommender.py

**Example Usage:**

*Hello there! Stuck on what to do with your life?*
*Or maybe you just want a change in career?*
*Take the questionnaire to find a career just for you!*

**Do you have a preferred field (e.g. medicine, social media, etc.)?**

data

**Please write 3–8 skills or interests:** 

python machine learning statistics

**Do you have any qualifications?:** 

BSc

**Write the lowest annual salary you are willing to accept:**

50000


**Your ideal jobs are:**

1. Data Analyst
   Minimum Salary: $55,000
   Maximum Salary: $85,000
   Match Score: 87.4321

Project Structure:

JobRecommenderProject/
├── your_script_name.py     # Main application
├── jobs_cache.pkl          # Auto-generated cache (speeds up future runs)
├── requirements.txt        # Required libraries
├── .gitignore              # Excludes large files from GitHub
└── README.md               # Project documentation


Caching:

On the first run, the program processes the full dataset and saves a cache file (jobs_cache.pkl). All future runs load from this cache, making them significantly faster!

Future Improvements:
 - Maybe building a web interface using Streamlit or Gradio
 - Adding location-based filtering
 - Allowing users to upload their own CV for auto-profiling
 

Thanks! 

-Fav

