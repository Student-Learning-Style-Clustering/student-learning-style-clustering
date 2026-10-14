# Student Learning-Style Clustering Based on Learning Behavior

TechCrush Cohort 8 AI/ML Engineer Certification Capstone.

## Run locally in VS Code

1. Install Python 3.11 or 3.12 and Git.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```

The app tries `student_performance.csv` or `merged_dataset.csv` in the project root first, then the public Zenodo dataset. You can also upload a CSV from the sidebar. The CSV must contain the 16 expected columns documented in `app.py`.

## GitHub collaboration

Create a repository inside your GitHub organisation, then push this project folder. Give active members repository access with an appropriate role. A suggested workflow is to let members work on separate branches and use pull requests to merge into `main`.

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build Streamlit student clustering app"
git branch -M main
git remote add origin https://github.com/YOUR-ORG/YOUR-REPO.git
git push -u origin main
```

Replace `YOUR-ORG/YOUR-REPO` with the actual organisation and repository. If GitHub asks you to authenticate, complete the browser sign-in or use GitHub CLI; do not put passwords or tokens in source code.

## Deploy to Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Select the workspace corresponding to the GitHub organisation that owns the repository.
3. Choose **Create app**.
4. Select the repository, branch `main`, and entrypoint `app.py`.
5. Deploy. Copy the generated `streamlit.app` URL for your group and examiner.

If the organisation's repository is not listed, check the GitHub/Streamlit authorisation and organisation third-party application access settings. The repository must include `requirements.txt`.

## Methodology note

K-Means uses six behavioural features only: StudyHours, Attendance, AssignmentCompletion, OnlineCourses, Discussions and EduTech. LearningStyle, ExamScore and FinalGrade are held out for post-clustering comparisons. Agglomerative clustering and DBSCAN comparisons use a reproducible sample to reduce cloud memory/time costs; explain this implementation detail in the presentation.
# student-learning-style-clustering
AI/ML Capstone Project - Student Learning-Style Clustering Behavioral Analysis
