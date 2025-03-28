import streamlit as st
import cv2
import pytesseract
import difflib
import re
import numpy as np
import os
import uuid
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from difflib import SequenceMatcher
from rapidfuzz import fuzz  # Faster fuzzy matching
from pdf2image import convert_from_path
import tempfile
from deepface import DeepFace
import tensorflow as tf
import traceback
import easyocr
import cv2
import re

# def save_uploaded_file(uploaded_file, folder="uploads"):
#     """Save uploaded file to a directory"""
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     file_path = os.path.join(folder, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return file_path

# def delete_all_files(folder="uploads"):
#     """Delete all files in the uploads folder"""
#     if os.path.exists(folder):
#         for file_name in os.listdir(folder):
#             file_path = os.path.join(folder, file_name)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)

# # Streamlit UI
# st.title("Document Upload Portal")
# st.write("Upload your Photo, National ID, and Certificate")

# # Upload fields
# photo = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
# national_id = st.file_uploader("Upload National ID", type=["jpg", "jpeg", "png", "pdf"])
# certificate = st.file_uploader("Upload Certificate", type=["jpg", "jpeg", "png", "pdf"])

# if st.button("Submit"):
#     if photo and national_id and certificate:
#         photo_path = save_uploaded_file(photo)
#         national_id_path = save_uploaded_file(national_id)
#         certificate_path = save_uploaded_file(certificate)
        
#         st.success("Files uploaded successfully!")
#         # st.image(photo_path, caption="Uploaded Photo", use_column_width=True)
#         # st.write(f"National ID saved at: {national_id_path}")
#         # st.write(f"Certificate saved at: {certificate_path}")
#     else:
#         st.warning("Please upload all required documents.")

# if st.button("Delete All Files"):
#     delete_all_files()
#     st.success("All uploaded files have been deleted.")
        

# Set TensorFlow logging to avoid warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set Tesseract path (if not in system PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set Poppler path for PDF to image conversion
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an application:", ["Face Comparison", "Certificate Comparison","NationalIds NameExtractor","Name Degree Checker","University Name Extractor"])

#Helper Function
#university certificate compare
def convert_pdf_to_image(pdf_path):
    """Convert the first page of a PDF to an image."""
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    if images:
        image_path = f"uploads/{uuid.uuid4().hex}.jpg"
        images[0].save(image_path, 'JPEG')
        return image_path
    return None

def extract_text_from_image(image_path):
    """Extract text from an image using OCR with PSM 6."""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def extract_university_name(text):
    """Extract university name by isolating lines with 'University/College/Institute'."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    university_candidates = []
    
    # Regex to match university names and trim trailing non-name text
    uni_pattern = re.compile(
        r"^(.*?(?:University|College|Institute).*?)(?:\s{2,}|\(|,|$)", 
        re.IGNORECASE
    )
    
    for line in lines:
        match = uni_pattern.search(line)
        if match:
            clean_name = match.group(1).strip()
            university_candidates.append(clean_name)
    
    return university_candidates[0] if university_candidates else None

def compare_university_names(uni1, uni2):
    """Compare similarity between two university names using text matching."""
    if not uni1 or not uni2:
        return 0.0
    return difflib.SequenceMatcher(None, uni1, uni2).ratio()

def image_ssim(image1_path, image2_path, region="all"):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print(f"Error loading images: {image1_path} or {image2_path}")
        return 0.0

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape

    if region == "top":
        region1 = gray1[:int(h * 0.2), :]
        region2 = gray2[:int(h * 0.2), :]
    elif region == "middle":
        region1 = gray1[int(h * 0.4):int(h * 0.6), :]
        region2 = gray2[int(h * 0.4):int(h * 0.6), :]
    elif region == "bottom":
        region1 = gray1[int(h * 0.8):, :]
        region2 = gray2[int(h * 0.8):, :]
    else:
        region1 = gray1
        region2 = gray2

    if region1.size == 0 or region2.size == 0:
        print("One of the regions is empty!")
        return 0.0

    region2 = cv2.resize(region2, (region1.shape[1], region1.shape[0]))

    region1 = img_as_float(region1)
    region2 = img_as_float(region2)

    score, _ = ssim(region1, region2, full=True, data_range=1.0)
    return round(score, 2)



def image_orb_similarity(image1_path, image2_path):
    """Computes ORB feature matching ratio."""
    orb = cv2.ORB_create()
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    if img1 is None or img2 is None:
        print(f"Error loading images for ORB: {image1_path} or {image2_path}")
        return 0.0

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0  # No keypoints found
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similar_regions = [m for m in matches if m.distance < 80]
    return len(similar_regions) / max(len(matches), 1)

def compare_text_sequence_exact(text1, text2):
    """Compare text sequences using SequenceMatcher for exact matching."""
    return SequenceMatcher(None, text1, text2).ratio()

def compare_text_sequence_fuzzy(text1, text2):
    """Compare text sequences using rapidfuzz for fuzzy matching."""
    return fuzz.partial_ratio(text1, text2) / 100.0

def preprocess_image(image_path):
    """Convert the certificate to grayscale and apply edge detection."""
    image = cv2.imread(image_path, 0)  # Load in grayscale
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    return edges, image

def extract_logo_regions(image):
    """Detect potential logo regions using contour detection."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logo_candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 50 < w < 500 and 50 < h < 500 and 0.3 < aspect_ratio < 3.0:
            logo_candidates.append((x, y, w, h))

    return logo_candidates

def orb_feature_matching(logo, candidate):
    """Use ORB feature matching to compare extracted logo with master logos."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(logo, None)
    kp2, des2 = orb.detectAndCompute(candidate, None)

    if des1 is None or des2 is None:
        return 0  # No keypoints found

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similar_regions = [m for m in matches if m.distance < 80]

    return len(similar_regions) / max(len(matches), 1)

def check_logo_in_certificate(master_logo_folder, certificate_path, threshold=0.4):
    """Compare detected regions in the certificate against master logos."""
    edges, certificate = preprocess_image(certificate_path)
    regions = extract_logo_regions(edges)

    best_match = 0
    best_logo = None

    for logo_file in os.listdir(master_logo_folder):
        logo_path = os.path.join(master_logo_folder, logo_file)
        logo = cv2.imread(logo_path, 0)  # Load master logo in grayscale

        for (x, y, w, h) in regions:
            extracted_region = certificate[y:y+h, x:x+w]  # Crop potential logo area

            if extracted_region.shape[0] > 0 and extracted_region.shape[1] > 0:
                score = orb_feature_matching(logo, extracted_region)
                if score > best_match:
                    best_match = score
                    best_logo = logo_file

                if score > threshold:  # Stop early if a good match is found
                    return best_logo, round(best_match * 100, 2)

    return best_logo, round(best_match * 100, 2)

def compare_certificates(image1_path, image2_path, master_logo_folder):
    """Compare two certificate images and return similarity score."""
    cert1_text = extract_text_from_image(image1_path)
    cert2_text = extract_text_from_image(image2_path)

    if not cert1_text or not cert2_text:
        return 0.0

    # Dynamically extract university names
    uni1 = extract_university_name(cert1_text)
    uni2 = extract_university_name(cert2_text)

    # Compare university names
    uni_similarity = compare_university_names(uni1, uni2)

    weights = {
        "university_header": 25,
        "degree_details": 5,
        "student_info": 5,
        "signatories_auth": 5,
        "layout_formatting": 10,
        "ssim_top_compare": 10,
        "ssim_bottom_compare": 10,
        "orb_compare": 20,
        "text_exact_sequence": 5,
        "text_fuzzy_sequence": 5,
    }

    similarity_score = 0
    # 1. University Name and Header Structure
    if uni_similarity >= 0.80:
        similarity_score += weights["university_header"] * 1.0
    elif uni1 and uni2:
        similarity_score += weights["university_header"] * uni_similarity

    # 2. Degree and Academic Details
    degree_keywords = ["Bachelor", "Bachelors", "Master", "Masters", "Doctor", "PhD.", "Diploma", "Undergraduate", "Postgraduate", "Graduate", "Honours", "Honors", "Distinction", "Merit", "Pass","Doctorate", "Doctor of Philosophy", "Doctor of Science", "Doctor of Medicine","MPhil", "MSc", "MA", "MEng", "MBA", "Executive MBA", "Postgraduate Diploma", "Postgraduate Certificate","Associate Degree", "Higher National Diploma", "HND","GPA", "CGPA", "Cumulative GPA", "Grade Point Average", "Percentage", "Marks Obtained","Academic Standing", "Credit Hours", "Completion Status", "First-Class", "Second-Class", "Third-Class","Subject:", "Major", "Minor", "Specialization", "Discipline", "Concentration", "Field of Study","Certified", "Completion", "Successfully Completed", "Fulfilled Requirements", "Accredited", "Recognized","Associate", "Summa Cum Laude", "Magna Cum Laude", "Cum Laude","Foundation Degree", "Ordinary Degree", "Taught Masters", "Research Masters", "Licence", "Maîtrise", "Diplôme", "Diplom", "Magister", "Licenciado", "Laurea", "Postgrado",
    "Shuoshi", "Xueshi", "Gakushi", "Kakushi", "Sarjana", "Magister", "Doktor", "Bachiller", "Licenciado", "Especialización", "Maestría", "Doctorado", "Ijazah", "Shahadat Al Ta'lim Al Aali", "Diplom", "Magister","National Diploma", "Higher Certificate", "Advanced Diploma"]
    cert1_degree = sum(1 for keyword in degree_keywords if keyword in cert1_text)
    cert2_degree = sum(1 for keyword in degree_keywords if keyword in cert2_text)
    degree_similarity = (cert1_degree + cert2_degree) / (2 * len(degree_keywords))
    similarity_score += weights["degree_details"] * degree_similarity

    # 3. Student and Parent Information
    student_match = difflib.SequenceMatcher(
        None,
        cert1_text.split("was awarded")[0] if "was awarded" in cert1_text else cert1_text,
        cert2_text.split("was awarded")[0] if "was awarded" in cert2_text else cert2_text
    ).ratio()
    similarity_score += weights["student_info"] * student_match

    # 4. Signatories and Authentication
    auth_keywords = ["Registrar", "Chancellor", "Signed", "Enrol. No.", "Rector", "Registration", "VC", "Vice Chancellor", "President"]
    cert1_auth = sum(1 for keyword in auth_keywords if keyword in cert1_text)
    cert2_auth = sum(1 for keyword in auth_keywords if keyword in cert2_text)
    auth_similarity = (cert1_auth + cert2_auth) / (2 * len(auth_keywords))
    similarity_score += weights["signatories_auth"] * auth_similarity

    # 5. Layout and Formatting
    layout_score = difflib.SequenceMatcher(
        None,
        re.sub(r"\s+", "", cert1_text),
        re.sub(r"\s+", "", cert2_text)
    ).ratio()
    similarity_score += weights["layout_formatting"] * layout_score

    # 6. SSIM Top Similarity
    ssim_top_score = image_ssim(image1_path, image2_path, region="top")
    similarity_score += weights["ssim_top_compare"] * ssim_top_score

    # 7. SSIM Bottom Similarity
    ssim_bottom_score = image_ssim(image1_path, image2_path, region="bottom")
    similarity_score += weights["ssim_bottom_compare"] * ssim_bottom_score

    # 8. ORB Similarity
    orb_score = image_orb_similarity(image1_path, image2_path)
    similarity_score += weights["orb_compare"] * orb_score

    # 9. Compare exact text sequence
    exact_match_score = compare_text_sequence_exact(cert1_text, cert2_text)
    similarity_score += weights["text_exact_sequence"] * exact_match_score

    # 10. Compare fuzzy text sequence
    fuzzy_match_score = compare_text_sequence_fuzzy(cert1_text, cert2_text)
    similarity_score += weights["text_fuzzy_sequence"] * fuzzy_match_score

    # 11. Master logo folder comparison
    logo_match, logo_score = check_logo_in_certificate(master_logo_folder, image1_path, threshold=0.4)
    if logo_score >= 40:
        similarity_score += logo_score * 0.1

    return round(similarity_score, 2)

def main():
    
    if option == "Certificate Comparison":
        
        st.title("Certificate Comparison Application")
        image1 = st.file_uploader("Upload the first certificate (image or PDF)", type=["png", "jpg", "jpeg", "pdf"])
        image2 = st.file_uploader("Upload the second certificate (image or PDF)", type=["png", "jpg", "jpeg", "pdf"])
        if image1 and image2:
            image1_path = f"uploads/{uuid.uuid4().hex}_{image1.name}"
            image2_path = f"uploads/{uuid.uuid4().hex}_{image2.name}"
            with open(image1_path, "wb") as f:
                f.write(image1.read())
            with open(image2_path, "wb") as f:
                f.write(image2.read())
            if image1.name.endswith(".pdf"):
                image1_path = convert_pdf_to_image(image1_path)
            if image2.name.endswith(".pdf"):
                image2_path = convert_pdf_to_image(image2_path)
            if image1_path and image2_path:
                master_logo_folder = r'C:\Users\inc3061\Documents\Master_Logo'
                similarity_score = compare_certificates(image1_path, image2_path, master_logo_folder)
                st.write(f"Similarity score between the certificates: {similarity_score}%")
                os.remove(image1_path)
                os.remove(image2_path)


            
if __name__ == '__main__':
    main()



# Function to compare two images
def compare_images(image1_path, image2_path):
    try:
        # Perform face verification using DeepFace with enforce_detection=False
        result = DeepFace.verify(image1_path, image2_path, enforce_detection=False)

        # Extract similarity score (distance)
        similarity_score = result['distance']

        # Check if the faces belong to the same person
        if result['verified']:
            message = "The faces belong to the same person."
        else:
            message = "The faces do not belong to the same person."

        return {
            "similarity_score": similarity_score,
            "verified": result['verified'],
            "message": message
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"An error occurred while processing the images: {e}"}

# Function to compare a PDF and an image
def compare_pdf_with_image(pdf_path, image_path):
    try:
        # Create a temporary folder to store the PDF page as an image
        with tempfile.TemporaryDirectory() as temp_folder:
            # Convert the PDF to images using Poppler
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

            # For simplicity, assume we compare only the first page of the PDF
            first_page_image_path = os.path.join(temp_folder, 'page_1.jpg')
            pages[0].save(first_page_image_path, 'JPEG')

            # Compare the first page of the PDF (converted to an image) with the provided image
            return compare_images(first_page_image_path, image_path)

    except Exception as e:
        traceback.print_exc()
        return {"error": f"An error occurred while processing the PDF: {e}"}

# Function to compare two PDFs
def compare_pdfs(pdf1_path, pdf2_path):
    try:
        # Create a temporary folder to store the PDF pages as images
        with tempfile.TemporaryDirectory() as temp_folder:
            # Convert the first PDF to images
            pages1 = convert_from_path(pdf1_path, dpi=300, poppler_path=poppler_path)
            first_page_pdf1_image_path = os.path.join(temp_folder, 'pdf1_page_1.jpg')
            pages1[0].save(first_page_pdf1_image_path, 'JPEG')

            # Convert the second PDF to images
            pages2 = convert_from_path(pdf2_path, dpi=300, poppler_path=poppler_path)
            first_page_pdf2_image_path = os.path.join(temp_folder, 'pdf2_page_1.jpg')
            pages2[0].save(first_page_pdf2_image_path, 'JPEG')

            # Compare the first page of both PDFs (converted to images)
            return compare_images(first_page_pdf1_image_path, first_page_pdf2_image_path)

    except Exception as e:
        traceback.print_exc()
        return {"error": f"An error occurred while processing the PDFs: {e}"}

# Streamlit frontend code
if option == "Face Comparison":
    
    st.title("Face Comparison App")
    st.write("Upload two images or PDFs to compare if the faces are of the same person.")
    
    # Upload image files
    uploaded_file1 = st.file_uploader("Upload the first image or PDF", type=['png', 'jpg', 'jpeg', 'pdf'])
    uploaded_file2 = st.file_uploader("Upload the second image or PDF", type=['png', 'jpg', 'jpeg', 'pdf'])
    
    if uploaded_file1 and uploaded_file2:
        # Save uploaded files temporarily
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
    
        image1_path = os.path.join('uploads', 'image1')
        image2_path = os.path.join('uploads', 'image2')
    
        with open(image1_path, 'wb') as f:
            f.write(uploaded_file1.getbuffer())
        
        with open(image2_path, 'wb') as f:
            f.write(uploaded_file2.getbuffer())
    
        # Check if both files are PDFs or images and call the appropriate function
        if uploaded_file1.name.lower().endswith('.pdf') and uploaded_file2.name.lower().endswith('.pdf'):
            result = compare_pdfs(image1_path, image2_path)
        elif uploaded_file1.name.lower().endswith('.pdf'):
            result = compare_pdf_with_image(image1_path, image2_path)
        elif uploaded_file2.name.lower().endswith('.pdf'):
            result = compare_pdf_with_image(image2_path, image1_path)
        else:
            result = compare_images(image1_path, image2_path)
    
        # Display the result
        if "error" in result:
            st.error(result["error"])
        else:
            st.write(f"Similarity Score (Distance): {result['similarity_score']}")
            st.write(result['message'])
    
    # If no files uploaded, display a message
    else:
        st.write("Please upload both files (images or PDFs) to proceed.")

# NationalIds NameExtractor
import os
import tempfile
from pdf2image import convert_from_path
import streamlit as st
from PIL import Image
import easyocr
import cv2
import re
# Specify the path to the Poppler 'bin' folder
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Adjust this path for your system

# Define supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make text clearer
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh

def extract_text_with_easyocr(image_path):
    """
    Extract text using EasyOCR from the image.
    """
    # Initialize the EasyOCR Reader (supports multiple languages)
    reader = easyocr.Reader(['en'])

    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform OCR using EasyOCR
    result = reader.readtext(image, detail=0)  # Set detail=0 to return just the text

    # Join the detected text
    extracted_text = "\n".join(result)

    return extracted_text

def parse_mrz_lines_with_criteria(mrz_text):
    """
    Extract lines that start with 'P' and contain one or more '<<' symbols.
    """
    # Split the text into lines
    lines = mrz_text.splitlines()

    # Filter lines that meet the criteria
    filtered_lines = [line for line in lines if line.startswith('P') and '<<' in line]

    return filtered_lines

def process_pdf(file_path, output_folder):
    """
    Convert a PDF to images and save them in the output folder.
    """
    # Convert the PDF to images
    pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)

    # Save each page as an image in the output folder
    image_paths = []
    for i, page in enumerate(pages):
        output_image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        page.save(output_image_path, 'JPEG')
        image_paths.append(output_image_path)

    return image_paths

def split_mrz_line(mrz_line, country_codes):
    """
    This function processes the MRZ line by extracting the document type, country code, given name,
    and the sentence after the country code. It checks if the country code exists in the provided list.
    """
    # Remove any special characters like * and -
    mrz_line = mrz_line.replace("*", "").replace("-", "")

    # Extract the type (first two characters)
    doc_type = mrz_line[:2]

    # Extract the country code (next three characters)
    country_code = mrz_line[2:5]

    # Extract the rest as the given name
    given_name = mrz_line[5:]

    # Find the sentence after the country code
    country_index = mrz_line.find(country_code)
    if country_index != -1:
        sentence_after_country_code = mrz_line[country_index + len(country_code):]
    else:
        sentence_after_country_code = ""

    # Check if the country code exists in the list of valid country codes
    country_exists = country_code in country_codes

    return sentence_after_country_code

def split_surname_given_name(sentence_after_country_code):
    """
    This function splits the sentence after the country code into surname and given name.
    The surname is the part before the first `<` and the given name is everything after it.
    If multiple `<` characters exist, the text between them is considered part of the given name.
    Additionally, any digits (numbers) will be removed from the names.
    """
    # Split the sentence by the first '<'
    parts = sentence_after_country_code.split('<', 1)

    # The part before the first '<' is the surname
    surname = parts[0].strip()

    # Remove any digits from the surname
    surname = re.sub(r'\d', '', surname)

    # The part after the first '<' is the given name (split by '<' and joined with space)
    given_name_part = parts[1].strip() if len(parts) > 1 else ""

    # Remove any digits from the given name part
    given_name_part = re.sub(r'\d', '', given_name_part)

    # Split by '<' to handle individual names separated by '<'
    given_name = ' '.join(given_name_part.split('<')).strip()

    return surname, given_name

# Function to clean and extract names
def extract_name(text, keywords, unwanted_words):
    # Convert the text to lowercase for case insensitivity
    text = text.lower()
    
    # Regular expression to remove unwanted characters (non-alphabetic)
    clean_text = re.sub(r'[^a-z\s]', '', text)
    
    result = {}
    
    # Iterate through the keywords to find names
    for keyword in keywords:
        # Create a regex pattern to capture the name after the keyword
        pattern = r'{}[\s:]*([a-z\s]+)'.format(re.escape(keyword.lower()))
        match = re.search(pattern, clean_text)
        
        if match:
            # Extract the name after the keyword
            name = match.group(1).strip()
            
            # Remove anything that isn't part of the name (numbers, special characters, etc.)
            name = re.sub(r'[^a-z\s]', '', name)
            
            # Limit name extraction to stop if we encounter unwanted words (like 'date', 'number', etc.)
            stop_words = ['date', 'number', 'place', 'village', 'signature', 'code']
            for word in stop_words:
                if word in name:
                    name = name.split(word)[0].strip()
                    break
            
            # Ensure we capture the first part of the name and stop there
            # Only retain the first two words of the name (for first name and surname)
            name_parts = name.split()
            if len(name_parts) > 2:
                name = ' '.join(name_parts[:2])  # Only capture the first two words
            
            # Store the result with the keyword as the key
            result[keyword] = ' '.join(name.split())
    
    # If no valid names are found, return "name not found"
    if not result:
        return "name not found"
    
    # Filtering unwanted words from the result
    filtered_result = {}
    for key, value in result.items():
        # Split the extracted name to check if it contains any unwanted words
        words = value.split()
        filtered_words = [word for word in words if word not in unwanted_words]
        
        # Join the filtered words back into a name
        filtered_name = ' '.join(filtered_words)
        
        # Store the filtered result if the filtered name is not empty
        if filtered_name:
            filtered_result[key] = filtered_name
    
    return filtered_result

# List of valid country codes
country_codes = [
    "AFG", "ALA", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "AUS", "AUT", "AZE",
    "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN", "BOL", "BIH", "BWA",
    "BRA", "BRN", "BGR", "BFA", "BDI", "KHM", "CMR", "CAN", "CPV", "CAF", "TCD", "CHL",
    "CHN", "COL", "COM", "COG", "COD", "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK",
    "DJI", "DMA", "DOM", "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI",
    "FIN", "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", "GNB",
    "GUY", "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL", "ISR", "ITA",
    "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT", "KGZ", "LAO", "LVA", "LBN", "LSO",
    "LBR", "LBY", "LIE", "LTU", "LUX", "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL",
    "MRT", "MUS", "MEX", "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM",
    "NRU", "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN", "PAK",
    "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "RWA",
    "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SGP",
    "SVK", "SVN", "SLB", "SOM", "ZAF", "KOR", "SSD", "ESP", "LKA", "SDN", "SUR", "SWE",
    "CHE", "SYR", "TWN", "TJK", "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR",
    "TKM", "TUV", "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VAT", "VEN",
    "VNM", "YEM", "ZMB", "ZWE"
]

# List of keywords to search for
keywords = ['surname', 'first name', 'first names', 'names', 'given names', 'full names', 'full name']

# List of unwanted words
unwanted_words = ['first', 'names', 'surname', 'name', 'full']

def main():
    # Streamlit user interface for uploading files
    if option == "NationalIds NameExtractor":
        
        st.title("Name Extractor Uploader")
        uploaded_file = st.file_uploader("Choose a file (PDF or Image)", type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'])
    
        if uploaded_file is not None:
            # Get file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
            # If the uploaded file is an image, apply similar layout as PDF processing
            if file_extension in image_extensions:
                # Create a temporary file for image storage
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_file_path = temp_file.name
    
                # Open the image
                image = Image.open(temp_file_path)
                st.write("Image file uploaded successfully.")
    
                # Resize the image to a smaller size (similar to PDF page size)
                image_resized = image.resize((300, 300))  # Adjust the size as needed
    
                # Use columns for better layout of image
                cols = st.columns(1)  # Single column for image
                with cols[0]:
                    st.image(image_resized, caption="Uploaded Image", use_container_width=True)
    
                # Perform OCR and extract MRZ lines
                extracted_text = extract_text_with_easyocr(temp_file_path)
                filtered_lines = parse_mrz_lines_with_criteria(extracted_text)
    
                # Display the results for the uploaded image
                with st.expander("OCR Results for Uploaded Image"):
                    if filtered_lines:
                        st.write("Filtered MRZ Lines:")
                        for line in filtered_lines:
                            st.write(line)
    
                        # Process the MRZ line and extract surname and given name
                        for line in filtered_lines:
                            sentence_after_country_code = split_mrz_line(line, country_codes)
                            surname, given_name = split_surname_given_name(sentence_after_country_code)
                            st.write(f"Surname: {surname}")
                            st.write(f"Given Name: {given_name}")
                    else:
                        st.write("No matching MRZ lines found. Attempting to extract names from text...")
                        # Extract names using keywords and unwanted words
                        name = extract_name(extracted_text, keywords, unwanted_words)
                        if isinstance(name, dict):
                            for key, value in name.items():
                                st.write(f"{key}: {value}")
                        else:
                            st.write(name)
    
            # If the uploaded file is a PDF, convert it to images and save them
            elif file_extension == '.pdf':
                st.write("Processing PDF... Please wait.")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                    temp_pdf_file.write(uploaded_file.getbuffer())
                    temp_pdf_file_path = temp_pdf_file.name
    
                with tempfile.TemporaryDirectory() as temp_output_dir:
                    # Convert PDF to images
                    image_paths = process_pdf(temp_pdf_file_path, temp_output_dir)
                    st.write(f"PDF processed. {len(image_paths)} pages converted to images.")
    
                    # Use columns for better layout of images
                    num_columns = 3  # Set the number of columns you want
                    cols = st.columns(num_columns)
                    
                    for i, image_path in enumerate(image_paths):
                        # Open image for displaying in Streamlit
                        page_image = Image.open(image_path)
                        col_idx = i % num_columns
                        with cols[col_idx]:
                            st.image(page_image, caption=f"Page {i+1} of the PDF", use_container_width=True)
                        
                        # Perform OCR and extract MRZ lines
                        extracted_text = extract_text_with_easyocr(image_path)
                        filtered_lines = parse_mrz_lines_with_criteria(extracted_text)
    
                        # Display the results for each page
                        with st.expander(f"Page {i+1} OCR Results"):
                            if filtered_lines:
                                st.write(f"Filtered MRZ Lines from Page {i+1}:")
                                for line in filtered_lines:
                                    st.write(line)
    
                                # Process the MRZ line and extract surname and given name
                                for line in filtered_lines:
                                    sentence_after_country_code = split_mrz_line(line, country_codes)
                                    surname, given_name = split_surname_given_name(sentence_after_country_code)
                                    st.write(f"Surname: {surname}")
                                    st.write(f"Given Name: {given_name}")
                            else:
                                st.write(f"No matching MRZ lines found on Page {i+1}. Attempting to extract names from text...")
                                # Extract names using keywords and unwanted words
                                name = extract_name(extracted_text, keywords, unwanted_words)
                                if isinstance(name, dict):
                                    for key, value in name.items():
                                        st.write(f"{key}: {value}")
                                else:
                                    st.write(name)

if __name__ == "__main__":
    main()

# Degree and Name Extractor 
# Set Poppler path (Poppler should be installed on your machine for PDF-to-image conversion)
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Update this to your Poppler path
os.environ["PATH"] += os.pathsep + poppler_path

# Set Tesseract OCR path (Windows users must specify the path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image):
    """Extract text from an image using OCR"""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray)  # Perform OCR
    return text.lower()  # Convert the text to lowercase

def extract_text_from_pdf(pdf):
    """Extract text from a PDF file by converting pages to images"""
    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf.read())
        temp_pdf_path = temp_pdf.name

    # Convert PDF to images using poppler
    pages = convert_from_path(temp_pdf_path, poppler_path=poppler_path)
    full_text = ""
    for page_num, page in enumerate(pages, start=1):
        img = np.array(page)
        text = pytesseract.image_to_string(img)
        full_text += f"Page {page_num}:\n{text}\n---\n"
    
    # Clean up the temporary file
    os.remove(temp_pdf_path)

    return full_text.lower()  # Convert the text to lowercase

def find_name_in_text(text, name):
    """Find the name in the extracted text and return its sequence & line"""
    lines = text.split("\n")  # Split text into lines
    words = text.split()  # Split text into words
    name_parts = name.lower().split()  # Convert the name to lowercase and split into individual words
    positions = []
    found_line = ""

    # Finding word sequence
    for i in range(len(words) - len(name_parts) + 1):
        if words[i:i + len(name_parts)] == name_parts:
            positions.append(i + 1)  # Convert to 1-based index

    # Finding the exact line where the name appears
    for line in lines:
        if name.lower() in line:  # Ensure both the text and name are in lowercase for comparison
            found_line = line.strip()
            break

    # Display results
    if positions:
        st.success(f"✅ Name '{name}' found in the document.")
        st.write(f"🔢 Word sequence position(s): {positions}")
        st.write(f"📌 Name appears in this line: \"{found_line}\"")
    else:
        st.error(f"❌ Name '{name}' not found in the document.")

def find_degree_lines(text):
    """Find lines containing degree patterns like Bachelor's, Master's, Doctorates"""
    degree_patterns = [
        r'\b(Bachelor|Master|Doctor|Diploma|Associate)\w*(?:\s*of\s*Arts|\s*Science|\s*Engineering|\s*Commerce)?\b',
        r'\b(B.A|B.Sc|B.Tech|M.A|M.Sc|M.Tech|Ph.D|M.B.A|M.C.A)\b',
        r'\bB\.Tech\b',
        r'\bM\.Tech\b',
        r'\bM\.A\b',
        r'\bPh\.D\b',
        r'b\.\w+',  # e.g., B.Sc, B.A
        r'M\.\w+',  # e.g., M.Sc, M.A
        r'\b(M\.S|M\.A|B\.S|B\.A|B\.Sc|M\.Sc|M\.Tech|B\.Tech|Ph\.D|M\.B\.A|M\.C\.A)\b[-\s]?[A-Za-z\s]+',
        r'\b(M\.S|M\.A|B\.S|B\.A|B\.Sc|M\.Sc|M\.Tech|B\.Tech|Ph\.D|M\.B\.A|M\.C\.A)\b',  # Matches degrees like M. S., M. A., etc.
        r'\bM\.\s*S\.\b',
        r'\bM\.\s*S\.\s*-\s*[A-Za-z]+\b'
    ]
    
    # Split text into lines
    lines = text.split('\n')
    found_lines = []
    
    for line in lines:
        for pattern in degree_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                found_lines.append(line)
                break  # No need to check other patterns once we find a match in the line
    
    return found_lines

def main():
    """Main function to process image or PDF and find the name sequence, line & degrees"""
    if option == "Name Degree Checker":
        st.title("Text Extraction from Images and PDFs")
        
        # Upload file
        uploaded_file = st.file_uploader("Choose a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])
        
        # Name input
        name_to_find = st.text_input("Enter the name to search for", "ntombifuthi")
    
        if uploaded_file is not None:
            # Determine file type
            if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                # If it's an image, process it directly
                extracted_text = extract_text_from_image(uploaded_file)
            elif uploaded_file.type == "application/pdf":
                # If it's a PDF, convert it to images first, then process
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload an image or PDF.")
                return
    
            # Find name in the extracted text
            find_name_in_text(extracted_text, name_to_find)
    
            # Find degrees in the extracted text
            degree_lines = find_degree_lines(extracted_text)
            if degree_lines:
                st.success("Degrees found in the document:")
                for line in degree_lines:
                    st.write(f"📜 {line}")
            else:
                st.warning("No degrees found in the document.")


if __name__ == "__main__":
    main()


# university name Extractor 

# Manually set the tesseract executable path (if it's not in the PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path if necessary

# Path to Poppler (used for PDF to image conversion)
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Adjust this path if necessary

# Function to extract text from an image using OCR with PSM 6 (single block of text)
def extract_text_from_image(image_path):
    """Extract text from an image using OCR with PSM 6."""
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # Perform OCR with PSM 6 (single uniform block of text)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

# Function to extract university name by isolating lines with 'University', 'College', or 'Institute'
def extract_university_name(text):
    """Extract university name by isolating lines with 'University/College/Institute'."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # List to hold the matching lines
    matching_lines = []
    
    # Iterate through each line and check if it contains any of the keywords
    for line in lines:
        if any(keyword in line for keyword in ['University', 'College', 'Institute']):
            matching_lines.append(line.strip())
    
    # If there are matching lines, return them; otherwise, return a message
    if matching_lines:
        return "\n".join(matching_lines)
    else:
        return "No lines with 'University', 'College', or 'Institute' found."

# Function to convert PDF to image and process the first page
def convert_pdf_to_image(pdf_path):
    """Convert the first page of a PDF to an image."""
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        if images:
            image_path = f"uploads/{uuid.uuid4().hex}.jpg"
            images[0].save(image_path, 'JPEG')
            return image_path
        return None
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None

# Function to process PDF and extract text from the first page
def extract_text_from_pdf(pdf_file):
    """Convert PDF to image and extract text from the first page."""
    try:
        # Create a temporary file path for the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_pdf_path = tmp_file.name
            tmp_file.write(pdf_file.read())  # Write the uploaded PDF to a temporary file
        
        # Convert PDF to image and get the image path
        image_path = convert_pdf_to_image(tmp_pdf_path)
        if image_path:
            # Perform OCR using the image
            return extract_text_from_image(image_path)
        else:
            return "Error: Could not convert PDF to image."
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return ""

# Streamlit UI for uploading an image or PDF
if option == "University Name Extractor":
    st.title("University Name Extractor")
    
    # Let the user upload an image or PDF
    uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file is not None:
        # Check if the uploaded file is a PDF
        if uploaded_file.type == "application/pdf":
            # Process the PDF and extract text from the first page
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            # If it's an image, process it as usual
            image = Image.open(uploaded_file)
            # Save the uploaded image to a temporary file to use with OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file_path = tmp_file.name
                image.save(tmp_file_path)
            # Perform OCR using the image
            extracted_text = extract_text_from_image(tmp_file_path)
        
        # Call the function to extract the university name from the text
        university_name = extract_university_name(extracted_text)
        
        # Display the result
        st.write("Extracted University Name:")
        st.write(university_name)







