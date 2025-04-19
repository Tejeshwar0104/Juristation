from flask import Flask, request, render_template, jsonify, send_file
import fitz
import re
import pandas as pd
import os
import spacy
from collections import defaultdict
from werkzeug.utils import secure_filename
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import BartForConditionalGeneration

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CSV for case retrieval functionality
existing_keywords_csv = "extracted_cases_with_links.csv"  # Update with actual CSV path
existing_df = pd.read_csv(existing_keywords_csv)

# Load NLP model for case analysis functionality
nlp = spacy.load("en_core_web_sm")

# Functions from the first app.py (Case retrieval)
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    text += doc[0].get_text("text")
    for page in doc[1:]:
        rect = page.rect
        new_rect = fitz.Rect(rect.x0, rect.y0 + 70, rect.x1, rect.y1)
        page.set_cropbox(new_rect)
        text += page.get_text("text")
    return text.replace("\n", " ").replace("\r", " ")

def split_into_paragraphs(text):
    paragraphs = text.split("\t")  # Splitting based on double newlines
    cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return cleaned_paragraphs  # Returning list of paragraphs

def find_relevant_paragraph(paragraphs, associated_words):
    for para in reversed(paragraphs):  # Iterate from last to first
        if any(word.lower() in para.lower() for word in associated_words):
            return para.split("Result of the")[0]  # Return the first match from the end
    return None

def extract_final_decision(text):
    # Directly assigned associated words for judgment, punishment, and conclusion
    associated_words = [
        # Judgment & Conclusion Keywords
        "judgment", "final decision", "court concludes", "held that", "verdict",
        "ordered", "directed", "declared", "decided", "ruled", "conclusion",
        "conclusively", "henceforth", "thus", "therefore", "it is concluded",

        # Criminal Punishment Keywords
        "sentenced to", "imprisonment", "death penalty", "life imprisonment",
        "rigorous imprisonment", "simple imprisonment", "conviction upheld",
        "punished with", "jail for", "custody for", "sentence shall be",

        # Acquittal Keywords
        "acquitted", "no punishment", "conviction set aside", "released from charges",
        "exonerated", "found not guilty",

        # Civil Case Outcomes
        "liable to pay", "ordered to pay", "shall pay", "compensation", "restitution",
        "financial obligation", "monetary penalty",

        # Penal Consequences for Non-Compliance
        "failure to pay", "liable for contempt", "may face penal consequences",
        "may be punished under CrPC", "legal consequences", "further action"
    ]

    return find_relevant_paragraph(text, associated_words)

def extract_keywords(text):
    match = re.search(r'List of Keywords (.+?) Case Arising Fr', text)
    if match:
        keywords_text = match.group(1).strip()
        keywords = [kw.strip() for kw in re.split(r'[;,\n]', keywords_text)]
        return keywords
    return ["Not Found"]

# Functions from the second app.py (Case analysis)
def extract_text_from_pdf_analysis(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_case_title(text):
  z=text[:8]
  found=False
  for ind,i in enumerate(z):
    if 'v.'==i.lower():
      index=ind
      found=True
      break
  if found:
    name=" ".join(text[(index-1):(index+2)])
    return name
  else:
    return text[1]


def extract_case_category(text):
  z=text[:7]
  for i in z:
    if 'Civil' in i:
      return 'Civil'
    elif 'Criminal' in i:
      return 'Criminal'
  return 'Writ Petition'

def extract_court_name(text):
    doc = nlp(text)
    court_name = 'Supreme Court of India'
    return court_name

def extract_judgment_date(text):
    match = re.findall(r'\b\d{1,2} \w+ \d{4}\b', text)
    return match[0] if match else "Not Found"

def extract_parties(text):
    # Regular expression to capture petitioner and respondent names
    match = re.search(r"\d+\s+S\.C\.R\..*?:\s+\d+\s+INSC\s+\d+\s+(.+?)\s+v\.\s+(.+?)\s+\(", text)

    if match:
        petitioner = match.group(1).strip()
        respondent = match.group(2).strip()
        return petitioner, respondent
    else:
        return None, None

def extract_judges(text):
    match = re.findall(r'\[(.*?)\]', text)
    return match[1] if match else "Not Found"

def extract_lawyers(text):
    pattern = (
        r"(?i)appearances\s*for\s*parties\s*(.*?)\s*"
        r"([\w\s\.,-]+?)\s*(?:advs?\.?|advocate|counsel)\s*for\s*the\s*(petitioners?|appellants?)\.?\s*"
        r"([\w\s\.,-]+?)\s*(?:advs?\.?|advocate|counsel)\s*for\s*the\s*(respondents?|respondent?)\.?\s*(.*?)(?:Judgment|$)"
    )

    match = re.search(pattern, text, re.DOTALL)

    if match:
        petitioner = match.group(2).strip().rstrip(",") + "." if match.group(2) else "Not Found."
        respondent = match.group(4).strip().rstrip(",") + "." if match.group(4) else "Not Found."
    else:
        pattern = r"(?i)appearances\s*for\s*parties\s*(.*?)\s*(?=judgment|$)"

        match = re.search(pattern, text, re.DOTALL)
        petitioner=match.group(0).split("Parties")[-1].split(", Adv")[0].strip().rstrip(",").rstrip(".")+"."
        para = match.group(0)
        if "etitioner" in para.lower():
            respondent = match.group(0).split("etitioner.")[-1].split(", Adv")[0].strip().rstrip(",").rstrip(".")+"."
        elif "etitioners" in para.lower():
            respondent = match.group(0).split("etitioners.")[-1].split(", Adv")[0].strip().rstrip(",").rstrip(".")+"."
        elif "ppellants" in para.lower():
            respondent = match.group(0).split("ppellants.")[-1].split(", Adv")[0].strip().rstrip(",").rstrip(".")+"."
        elif "ppellant" in para.lower():
            respondent = match.group(0).split("ppellant.")[-1].split(", Adv")[0].strip().rstrip(",").rstrip(".")+"."
        else:
            respondent="Not present."

        if len(respondent)<5:
            respondent="Not Present."
    return petitioner, respondent

def extract_acts(text):
    acts_match = re.search(r"(?<=List of Acts\s)(.*?)(?=\sList)", text, re.DOTALL)
    acts_text = acts_match.group(0).strip() if acts_match else ""

    # Split acts using ';' separator and remove extra spaces
    acts_list = [act.strip() for act in acts_text.split(';') if act.strip()]

    return acts_list

def extract_case_laws(text):
    match = re.search(r"Case Law Cited\s+(.+?)\s+List", text, re.DOTALL)

    if match:
        cases_text = match.group(1).strip()
        # Split case laws based on known delimiters like ";", "– referred to.", "– distinguished."
        cases = re.split(r";|\s–\sreferred\s+to\.|\s–\sdistinguished\.|\s–\srelied\s+on\.", cases_text)
        cases = [case.strip() for case in cases if case.strip()]  # Clean up spaces
        return cases
    else:
        return []

def extract_detected_crimes(text):
    crime_mapping = {
        "domestic violence": "Domestic Violence / Cruelty (IPC 498A, Domestic Violence Act)",
        "murder": "Murder (IPC 302)",
        "assault": "Physical Assault (IPC 354, IPC 323)",
        "fraud": "Cheating / Fraud (IPC 420)",
        "theft": "Theft (IPC 378, IPC 379)",
        "dowry": "Dowry Harassment (Dowry Prohibition Act, IPC 498A)",
        "harassment": "Sexual Harassment / Stalking (IPC 354A, IPC 509)",
        "extortion": "Extortion (IPC 383, IPC 384)",
        "rape": "Rape (IPC 376)",
        "kidnapping": "Kidnapping / Abduction (IPC 363, IPC 364A)",
        "bribery": "Bribery / Corruption (Prevention of Corruption Act, IPC 171B)",
        "misuse of funds": "Criminal Breach of Trust (IPC 406, IPC 409)",
        "money laundering": "Money Laundering (Prevention of Money Laundering Act, IPC 403)",
        "drug trafficking": "Drug Trafficking (Narcotic Drugs and Psychotropic Substances Act)",
        "cybercrime": "Cybercrime / Hacking / Identity Theft (IT Act 2000, IPC 66C, IPC 66D)",
        "torture": "Domestic Violence / Cruelty (IPC 498A)",
        "mental agony": "Harassment / Mental Cruelty (IPC 498A)",
        "abuse": "Verbal / Physical Abuse (IPC 294, IPC 323)",
        "human trafficking": "Human Trafficking (IPC 370, IPC 372)",
        "corruption": "Corruption / Misuse of Power (Prevention of Corruption Act, IPC 13)"
    }

    ipc_sections = {
        "498A": "Cruelty by Husband or Relatives",
        "302": "Murder",
        "307": "Attempt to Murder",
        "376": "Rape",
        "354": "Assault on a Woman",
        "509": "Sexual Harassment",
        "420": "Cheating and Fraud",
        "406": "Criminal Breach of Trust",
        "506": "Criminal Intimidation",
        "120B": "Criminal Conspiracy",
        "295A": "Blasphemy",
        "392": "Robbery",
        "395": "Dacoity",
        "363": "Kidnapping",
        "383": "Extortion",
    }

    crime_keywords = ["domestic violence","murder", "assault", "fraud", "theft", "dowry",
                      "domestic violence","harassment", "extortion", "rape", "kidnapping",
                      "bribery", "misuse of funds","money laundering", "drug trafficking",
                      "cybercrime", "torture", "mental agony","abuse", "human trafficking",
                      "corruption"]

    found_ipc = []
    for section in ipc_sections.keys():
        if f"section {section}" in text.lower() or f"sec {section}" in text.lower():
            found_ipc.append((section, ipc_sections[section]))

    # Extract crime keywords from text
    found_crimes = [word for word in crime_keywords if word in text.lower()]

    # Map crimes to legal descriptions using crime_mapping dictionary
    mapped_crimes = set([crime_mapping.get(crime, crime) for crime in found_crimes])

    # Add IPC-based crimes (already in formatted descriptions)
    mapped_crimes.update([desc for _, desc in found_ipc])

    return ', '.join(mapped_crimes) if mapped_crimes else 'No explicit crime detected'

def extract_verdict(text):
  x=" ".join(text.split("\t")[-5:])
  pattern = r"Result of the case:\s*(.*?)(?:\s*†Headnotes|\s*$)"
  match = re.search(pattern, x, re.DOTALL)
  if match:
      return match.group(1).strip()
  else:
      pattern = r"Result of the Case:\s*(.*?)(?:\s*†Headnotes|\s*$)"
      match = re.search(pattern, text, re.DOTALL)
      if match:
          return match.group(1).strip()
      else:
          return "Not Found"

def extract_punishment(text):
    match = re.search(r'(sentenced to|imprisonment|death penalty|jail for).*?', text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else "No explicit punishment mentioned."

def extract_evidence(text):
    match = re.search(r'(exhibit|documentary evidence|witness statement|CCTV footage).*?', text, re.IGNORECASE)
    return match.group(0).strip() if match else "No specific evidence mentioned."


def extract_context(text):
    # Extract the portion between "Issues for Consideration" and "Headnotes"
    issues_match = re.search(r"(?<=Issue for Consideration\s)(.*?)(?=\sHead)", text, re.DOTALL)
    issues_text = issues_match.group(0).strip() if issues_match else ""

    return issues_text

# from transformers import AutoTokenizer, BartForConditionalGeneration

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# model.eval() # Evaluate mode to avoid training

# def generate_summary(text, max_length):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=1024)
#     summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary


# Routes
@app.route('/')
def index():
    return render_template("home.html")

# Routes from second app.py (Case analysis)
@app.route('/case_analysis', methods=['GET', 'POST'])
def case_analysis():
    if request.method == 'POST':
        file = request.files['pdf']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            text = extract_text_from_pdf(filepath)
            paragraphs = split_into_paragraphs(text)
            text=text.replace("\n"," ")

            x=paragraphs[0].split("\n")
            text2=[]
            for i in x:
                text2.append(i.strip())

            raw=paragraphs[1:]
            judgement_text=[]
            for j in raw:
                judgement_text.append(j.replace("\n"," "))
            
            
            case_category = extract_case_category(text2)
            court_name = extract_court_name(text)
            judgment_date = extract_judgment_date(text)
            petitioner, respondent = extract_parties(text)
            judges_names = extract_judges(text)
            appellant_lawyer, respondent_lawyer = extract_lawyers(text)
            acts = extract_acts(text)
            case_laws = extract_case_laws(text)
            detected_crimes = extract_detected_crimes(text)
            verdict = extract_verdict(text)
            judgement=extract_final_decision(judgement_text)
            punishment = extract_punishment(text)
            evidence = extract_evidence(text)
            case_number = petitioner + ' ' + 'Vs' + ' ' + respondent
            context = extract_context(text)
            # summary = generate_summary(context, max_length=(len(context.split())+50))
            
            return render_template('case_analysis.html', case_number=case_number, case_category=case_category, court_name=court_name,
                                   judgment_date=judgment_date, petitioner=petitioner, respondent=respondent,
                                   judges_names=judges_names, appellant_lawyer=appellant_lawyer,
                                   respondent_lawyer=respondent_lawyer,
                                   acts=acts, case_laws=case_laws, judgement=judgement, detected_crimes=detected_crimes,
                                   verdict=verdict, punishment=punishment, evidence=evidence)
    
    return render_template('case_analysis.html')


# Routes from first app.py (Case retrieval)
@app.route('/case_retrieval')
def case_retrieval():
    return render_template("case_retrieval.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    new_text = extract_text_from_pdf(file_path)
    new_keywords = set(kw.lower().strip() for kw in extract_keywords(new_text))
    
    pdf_ranking = []
    for _, row in existing_df.iterrows():
        pdf_name = row["PDF Name"]
        summary = row['Summary']
        caseno = row['Case Title']
        link_column = row['Document Link']
        existing_keywords = set(kw.strip().lower() for kw in re.split(r'[;,\n]', str(row["Keywords"])) if kw.strip())
        intersection = new_keywords.intersection(existing_keywords)
        match_count = len(intersection)
        if match_count > 0:
            pdf_ranking.append({
                "caseno": caseno,  # Match frontend key
                "title": pdf_name,  # Title (frontend expects case title)
                "summary": summary,
                "link_column": link_column,
                'count': match_count
            })

    data_sorted = sorted(pdf_ranking, key=lambda x: x["count"], reverse=True)
    return jsonify({"results": data_sorted})

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Railway's PORT env variable
    app.run(host='0.0.0.0', port=port)
