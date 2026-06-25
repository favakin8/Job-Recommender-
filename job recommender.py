import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test")  
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def lemmatize_text(text):
    if pd.isnull(text):
        return ""
    words = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()
    return ' '.join(lemmatizer.lemmatize(w) for w in words)

unwanted_types = ('Part-Time|Intern|Contract|Temporary')
features= ['Qualifications', 'Job Title','Job Description', 'skills', 'Responsibilities']
synonyms = {
    # synonyms
    'medicine': 'medical healthcare clinical physician nursing doctor surgery hospital patient care',
    'law': 'legal attorney lawyer paralegal compliance regulatory litigation contract',
    'education': 'teaching teacher tutor academic curriculum school university lecturer training',
    'engineering': 'engineer technical mechanical electrical systems infrastructure design',
    'construction': 'civil structural building architecture site surveying planning',
    'coding': 'programming software development python java javascript developer',
    'finance': 'financial accounting banking investment analyst auditing budgeting',
    'social media': 'digital marketing content creator instagram influencer brand engagement',
    'marketing': 'advertising brand campaign market research SEO copywriting promotions',
    'sales': 'business development revenue account management client acquisition CRM',
    'design': 'graphic visual UI UX creative Adobe Figma branding illustration',
    'science': 'research laboratory biology chemistry physics data analysis experimentation',
    'environment': 'sustainability ecology conservation climate renewable energy green',
    'logistics': 'supply chain operations warehouse distribution procurement inventory',
    'security': 'cybersecurity network firewall threat analysis penetration testing',
    'hr': 'human resources recruitment talent management payroll employee relations',
    'data': 'data science analytics machine learning SQL statistics modelling visualisation',
    'writing': 'content copywriting editorial journalism author communication publishing',
    'hospitality': 'hotel restaurant tourism food service customer experience catering',
    'real estate': 'property estate agent housing mortgage valuation surveying',
    'art': 'creative illustration photography video animation visual media',
    'government': 'public sector policy administration civil service regulatory affairs',
    'nonprofit': 'charity voluntary social impact fundraising NGO community outreach',
    'consulting': 'strategy advisory management business analyst problem solving stakeholder',
    'psychology': 'counselling mental health therapy behavioural cognitive clinical social work',
    'bsc': 'bachelor science undergraduate degree',
    'ba': 'bachelor arts undergraduate degree',
    'msc': 'master science postgraduate degree',
    'ma': 'master arts postgraduate degree',
    'mba': 'master business administration management executive',
    'phd': 'doctorate research academic professor scientific',
    'no': 'entry level junior graduate trainee assistant',  # no qualifications = entry level roles
}
def clean(jobs):
    jobs=jobs.drop(['Role','Experience','latitude', 'location','Country',  'longitude','Company Size','Job Posting Date', 'Preference', 'Contact Person', 'Contact','Job Portal','Benefits','Company', 'Company Profile'],axis=1)
    jobs[['Min Salary','Max Salary']]=jobs['Salary Range'].str.split('-', expand=True)
    jobs['Min Salary'] = (jobs['Min Salary'].str.replace(r'[\$,]', '', regex=True).str.replace(r'[kK]', '000', regex=True).str.extract(r'(\d+)', expand=False).astype(float))
    jobs['Max Salary'] = (jobs['Max Salary'].str.replace(r'[\$,]', '', regex=True).str.replace(r'[kK]', '000', regex=True).str.extract(r'(\d+)', expand=False).astype(float))
    jobs = jobs[~jobs['Work Type'].str.contains(unwanted_types, case=False, na=False)]
    return jobs
CACHE_FILE = 'jobs_cache.pkl'
if os.path.exists(CACHE_FILE):
    print("Loading cached data...")
    with open(CACHE_FILE, 'rb') as f:
        jobs, vectorizer, x = pickle.load(f)
else:
    print("Processing data for first time (this takes a while)...")
    jobs = pd.read_csv('job_descriptions.csv')
    jobs= clean(jobs)
    for col in features:
        jobs[col] = jobs[col].astype(str).apply(lemmatize_text)
    jobs['combined'] = jobs[features].fillna('').apply(lambda row: ' '.join(row), axis=1)
    vectorizer=TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=6000, lowercase=True)
    x=vectorizer.fit_transform(jobs['combined'])
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump((jobs, vectorizer, x), f)
    print("Data cached! Future runs will be instant.")

def expand_synonyms(text):
    for word, expansion in synonyms.items():
        if word in text.lower():
            text += ' ' + expansion
    return text

def questions():
    field = input("Do you have a preferred field (e.g. medicine, social media, etc.)\n'" \
    "'If not, write the exact word 'no': ")
    if field.strip().lower() in ['no', 'none', 'n/a']:
        print("That's alright! Let's keep digging.")
    skill = input("Please write 3–8 skills or interests (e.g. python, social media, writing):\n'" \
    "'(The more you write, the better your matches will be.)\n")
    qualif = input("Do you have any qualifications (e.g. BA, BSc, MSc)? Or what qualification are you willing to get?\n'" \
    "'Please write the abbreviated version (e.g. MSc). If you are unsure, write 'no': ")
    pay = input("Write the **lowest annual salary** you are willing to accept. Be realistic :) ")
    return field, skill, qualif, pay
# Cleaning
def clean_salary(pay):
    if isinstance(pay, (int, float)):
        return int(pay)
    pay = re.sub(r'[\$,]', '', pay.strip())
    pay = re.sub(r'[kK]', '000', pay)
    match = re.search(r'[\d.]+', pay)
    if not match:
        raise ValueError("Please enter a valid number (e.g. 50000 or 50k)")
    return int(float(match.group()))

def clean_text(field, skill, qualif):
    no_responses = ['no', 'none', 'n/a']
    field_blank = field.strip().lower() in no_responses
    qualif_blank = qualif.strip().lower() in no_responses

    if field_blank and qualif_blank:
        userprofile = skill
    elif field_blank:
        userprofile = f"{skill} {qualif}"
    elif qualif_blank:
        userprofile = f"{field} {skill}"
    else:
        userprofile = f"{field} {skill} {qualif}"
    userprofile = re.sub(r'[^a-z0-9\s]', '', userprofile.lower())
    userprofile = re.sub(r'\s+', ' ', userprofile).strip()
    words = userprofile.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    if len(words) < 4:
        words += ['job', 'career', 'role', 'opportunity', 'work']
    return ' '.join(words)


mainloop=True
while mainloop:
    ##questionnaire
    print('Hello there! Stuck on what to do with your life?"\n' \
    'Or maybe you just want a change in career? \n' \
    'Take the questionnaire to find a career just for you!')
    # Questionnaire function
    while True:
        field, skill, qualif, pay = questions()
        pay = clean_salary(pay)
        userprofile = clean_text(field, skill, qualif)

        print("\nProfile:", userprofile)
        print("Minimum salary:", pay)
        yn = input("Is this correct? (y/n): ").strip().lower()
        if yn == 'y':
            break
        elif yn == 'n':
            print("Let's try again.\n")
        else:
            print("Please enter 'y' or 'n'.")

    userprofile = clean_text(field, skill, qualif)
    userprofile = expand_synonyms(userprofile)
    profile = vectorizer.transform([userprofile])

    similarity=cosine_similarity(profile,x).flatten()
    jobs['similarity']=similarity
    max_sal = jobs['Max Salary'].max()
    jobs['salary_score'] = jobs['Max Salary'] / max_sal 
    jobs['final_score'] = (0.70 * jobs['similarity']) + (0.30 * jobs['salary_score'])

    recommended = jobs.sort_values(by='final_score', ascending=False)
    bestjobs = recommended[recommended['similarity'] > 0.01].head(50)

    # salaryfilter
    filtered = bestjobs[bestjobs['Min Salary'] >= pay]
    filtered = filtered.drop_duplicates(subset=['Job Title'], keep='first')
    filtered = filtered.drop_duplicates(subset=['Job Description'], keep='first')


    if filtered.empty:
        print("Hmm, no jobs matched your salary minimum. Let's see if there are others to your match.")
        filtered = bestjobs.head(10)
    if filtered.empty:
        print("Sorry, no jobs matched your criteria.")
    else:
        print("Your ideal jobs are: ")
        for i, (_, item) in enumerate(filtered.iterrows(), start=1):
            print(f"\n{i}. {item['Job Title']}")
            print(f"   Minimum Salary: ${item['Min Salary']:.0f}")
            print(f"   Maximum Salary: ${item['Max Salary']:.0f}")
            print(f"   Description: {item['Job Description']}")
            print(f"   Skills: {item['skills']}")
            print(f"   ID: {item['Job Id']}")
            print(f"   Match Score: {item['final_score']*100:.4f}")
    again = input("\nTry again? (y/n): ").strip().lower()
    if again != 'y':
        print("Thanks for using the career finder!")
        break
