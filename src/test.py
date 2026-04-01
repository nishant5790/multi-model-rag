from unstructured.partition.pdf import partition_pdf

raw_pdf_elements=partition_pdf(
    filename="/Users/kumarnishant/DexterLab/multi-modal-RAG/data/Annual-Report-Analysis.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,
    extract_image_block_types=["Image","Table"],
    extract_image_block_to_payload=False,
    extract_image_block_output_dir="extracted_data"
    )


Header=[]
Footer=[]
Title=[]
NarrativeText=[]
Text=[]
ListItem=[]
Image=[]
Table=[]
for element in raw_pdf_elements:
  if "unstructured.documents.elements.Header" in str(type(element)):
            Header.append(str(element))
  elif "unstructured.documents.elements.Footer" in str(type(element)):
            Footer.append(str(element))
  elif "unstructured.documents.elements.Title" in str(type(element)):
            Title.append(str(element))
  elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
            NarrativeText.append(str(element))
  elif "unstructured.documents.elements.Text" in str(type(element)):
            Text.append(str(element))
  elif "unstructured.documents.elements.ListItem" in str(type(element)):
            ListItem.append(str(element))
  elif "unstructured.documents.elements.Image" in str(type(element)):
            Image.append(str(element))
  elif "unstructured.documents.elements.Table" in str(type(element)):
            Table.append(str(element))

print(Header)