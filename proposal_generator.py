#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymupdf
import easyocr
from PIL import Image, ImageEnhance
import io
from io import BytesIO
import cv2
import numpy as np
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, ListItem, ListFlowable, SimpleDocTemplate
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import streamlit as st
load_dotenv()


# In[13]:


class SOW_summary(BaseModel): 
    #subject : Optional[str] = Field(default="none") 
    subject: str
    tasks : str 
    total_cost : float


# In[5]:


def extract_images_from_pdf(uploaded_file):
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                # Convert to PIL Image object
                img = Image.open(io.BytesIO(image_bytes))
                #enhance the image
                enhancer = ImageEnhance.Contrast(img)
                high_contrast_img = enhancer.enhance(2)
                #prepare for easyocr by converting to array
                img_np = np.array(high_contrast_img)
                if img_np.ndim == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                extracted_images.append(img_np)
        doc.close()
        return extracted_images

def ocr_images_with_easyocr(image_list):
    reader = easyocr.Reader(['en']) # Specify languages, e.g., 'en' for English
    ocr_results = []
    for img in image_list:
        # EasyOCR can directly process PIL Image objects
        results = reader.readtext(img)
        ocr_results.append(results)
    return ocr_results



def create_proposal(file, buffer, percentage, project):

    # Extract images
    images = extract_images_from_pdf(file)

    # Perform OCR on extracted images
    all_ocr_results = ocr_images_with_easyocr(images)

    context_for_llm = " ".join([item[1] for item in all_ocr_results[0]])
    word_count = len(context_for_llm.split())

    #make llm request

    client = instructor.from_provider(
        "openrouter/openai/gpt-4o-mini",  # choose your model
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = """
    Summarize the following statement of work.  Please respond only with a valid JSON object using the following exact fields:
    - subject (string): A concise description of the work.
    - tasks (string): Extract the details of each task that will be done.   You MUST seperate each task with |.  
    - total_cost (float): Provide the total cost. 
    Return the output as valid JSON in the specified schema.  
    """

    statement_of_work = context_for_llm

    resp = client.chat.completions.create(
        messages = [{"role": "user", "content": prompt + statement_of_work}],
        response_model = SOW_summary,
        extra_body={"provider": {"require_parameters": True}},
    )
    fields = list(resp.model_dump().values())

    #get text
    subject = fields[0]
    tasks = fields[1]
    task_list = [part for part in tasks.split('|')]
    total_cost = fields[2]
    mark_up = 1.07
    total_cost = round( total_cost * (percentage + 1), 2)

    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    right_aligned_style = ParagraphStyle(
        name='RightAligned',
        parent=styleN,
        alignment=TA_RIGHT
    )
    title_style = ParagraphStyle(
        'LargeTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=14,  # Specify the font size here
        leading=50,   # Adjust leading for large fonts
        alignment=1   # Center alignment (0=left, 1=center, 2=right)
    )

    def letterhead(canvas, doc):
        scale = .9
        logo_size = 80
        canvas.saveState()
        # Draw logo
        canvas.drawImage(
            'https://github.com/jjemba83-lgtm/proposal_generator/blob/main/iet_logo_only.JPG?raw=true',
            72, doc.height + doc.topMargin - 60, width = scale * logo_size, height= logo_size
        )
    
        # Organization Name and Address
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(160, doc.height + doc.topMargin - 15, "Industrial Electric & Testing")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(160, doc.height + doc.topMargin - 30, "P.O. Box 2816, Tulsa, OK 74101")
        canvas.drawString(160, doc.height + doc.topMargin - 45, "Phone: (918) 592-6560 | www.ietOK.com")
        canvas.restoreState()

    doc = BaseDocTemplate(buffer, pagesize=letter)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 1*inch, id='normal')
    template = PageTemplate(id='letterhead', frames=frame, onPage=letterhead)
    doc.addPageTemplates([template])
    pn = project 
    current_date = datetime.now().strftime("%B %d, %Y")
    recipient_name = 'Susan Gustafson'
    recipient_org = 'Lumen Vyrex Operations'
    recipient_email = 'susan.gustafson@lumen.com'
    recipient_address = '100 S. Cincinnati Ave, Suite 1200 <br/> Tulsa OK'

    story = []
    story.append(Spacer(1, .2 * inch))
    story.append(Paragraph("PROPOSAL", title_style))
    story.append(Spacer(1, .1 * inch))
    story.append(Paragraph(f"Date: {current_date}", styleN))
    story.append(Paragraph(f"Quotation #: <u>{pn}</u>", right_aligned_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"{recipient_name}<br/>{recipient_org}<br/>{recipient_address}<br/>{recipient_email}", styleN))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"RE: {subject}", styleN))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Dear {recipient_name},<br/><br/>In response to your request, Industrial Electric & Testing is pleased to submit the following: ", styleN))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Scope:"))
    # Create Paragraphs and ListItems in a loop
    bullets = []
    for task in task_list:
        p = Paragraph(task, styleN)
        item = ListItem(p, leftIndent=30)  # Set leftIndent as needed for bullet+text
        bullets.append(item)

    bulleted_list = ListFlowable(
        bullets,
        bulletType="bullet",
        leftIndent=20  # Adjust to indent whole list block
    )
    story.append(bulleted_list)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"<b>Price: <u>${total_cost}</u></b>", styleN))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("This quote does not include any additional charges for taxes, fees, or permits.<br/>Industrial Electric & Testing Co. appreciates the opportunity to serve you. If you have any question concerning this quotation, please feel free to call me at (918) 592-6560.", styleN))   
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Sincerely", styleN))
    doc.build(story)
    return doc


# In[ ]:


#run the code
def main():
    st.title("Proposal Generator")
    project_number = st.text_input("Enter project number", value = "9009-")
    markup = st.number_input("Insert a markup:", value = 0.0)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    #if uploaded_file is not None:
        #pdf_bytes = uploaded_file.read()
    # User triggers PDF generation (e.g., button click)
    if st.button("Generate PDF"):
        # Your PDF generation code
        pdf_buffer = BytesIO()
        #doc = SimpleDocTemplate(pdf_buffer)
        result = create_proposal(uploaded_file, pdf_buffer, markup, project_number)  # your data for PDF
        #doc.build(result)
        pdf_buffer.seek(0)
        #st.pdf(pdf_buffer) #Not workign on the cloud for some reason
        # Flag that PDF is ready
        pdf_ready = True
    else:
        pdf_ready = False

    if pdf_ready:
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="proposal.pdf",
            mime="application/pdf"
        )
if __name__ == "__main__":
    main()


