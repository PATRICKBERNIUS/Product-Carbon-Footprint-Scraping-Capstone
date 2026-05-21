import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
import fitz, io
import pdfplumber
from pypdf import PdfReader
import anthropic
import json
import base64
from portkey_ai import Portkey
from portkeyfunc import _PortkeyCompat
from openai import AzureOpenAI
from tqdm import tqdm
import time




#Azure OpenAI key information
endpoint = "use own endpoint"
model_name = "gpt-5.2-chat"
deployment = "gpt-5.2-chat"


subscription_key = "use own key"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)






# -------- DELL -------- #

def get_dell_soup():
    url = "https://www.dell.com/en-uk/lp/dt/product-carbon-footprints#Desktops" #dell url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
    } #headers to bypass blocking
 
    req = requests.get(url, headers=headers) #using requests to grab html
 
    req_h = req.text #extracting the text
 
    soup = BeautifulSoup(req_h, "html.parser") #parsing the html
 
    return soup #returns parsed html





#function for extracting information from pdfs
def get_text_from_dell_pdf(soup, old_pdfs=None):

    full_dict = {} #empty dictionary to store all information

    s = soup.find_all("a", href=re.compile(r'pcf-report\.pdf|pcf-datasheet\.pdf')) #within the html, find all links for pdfs 

    
    old_pdfs = set(old_pdfs) if old_pdfs is not None else set() #ensure old_pdfs is a set

    pdfs = [link['href'] for link in s if link['href'] not in old_pdfs] #list containing all new pdfs

    failures = [] #empty list for storing failures


    for pdf_link in tqdm(pdfs, "Fetching Dell PCF Information"): #loop through all pdfs
        try:
            pdf = requests.get(pdf_link) #requests pdf link
            pdf.raise_for_status() #raise error if download failed
            pdf_bytes = pdf.content #convert pdf to bytes


            # Convert PDF to images
            # Use PyMuPDF to store 
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            image_content = [] #list to store image content

            for page_num in range(len(doc)): #loop through each page of pdf
                page = doc[page_num] #grab current page sotred doc
                pix = page.get_pixmap(dpi=200) #render the page
                img_bytes = pix.tobytes('png') #convert the rendered page to png bytes
                img_base64 = base64.b64encode(img_bytes).decode('utf-8') #encode as base64 string

                # Add each page as an image block
                image_content.append({
                    "type": "image_url", #specifc to chatgpt
                    "image_url": f"data:image/png;base64,{img_base64}"
                
                })

            doc.close() #free pdf from memory once all pages have been processed

            #create prompt including image content and directions
            message_content = image_content + [{
                "type": "text",
                    "text": """Extract the following data from this Dell product carbon footprint PDF and return ONLY a JSON object with these exact keys:

        {
        "model_name": "full product name as shown in the document header (e.g., 'Latitude 7350' or 'Dell Chromebook 3110')",
        "carbon_footprint": "number only (e.g., 214)",
        "manufacturing": "percentage only (e.g., 52.8)",
        "transportation": "percentage only (e.g., 4.7)",
        "use": "percentage only (e.g., 39.4)",
        "eol": "percentage only (e.g., 2.8)",
        "product_lifetime": "number only in years (e.g., 4)",
        "product_weight": "number only in kg (e.g., 5.45)"
        }

        ===== STEP 1: EXTRACT BASE VALUES =====

        model_name:
        - Find the product name in the large heading near the top of the first page (e.g., 'Dell Tower ECT1250', 'Latitude 7350', 'Dell Chromebook 3110')

        carbon_footprint:
        - Look for 'Total Carbon Footprint' — it is usually displayed as a large bold number followed by 'kg CO2-eq'
        - If shown with a regional label (e.g., 'EU Baseline'), use that value
        - If shown as a mean with uncertainty range (e.g., '229 +/- 42 kgCO2e'), use the mean value only
        - Always use the value that EXCLUDES end-of-life credits

        product_lifetime:
        - Find in the assumptions table, usually labeled 'Product Lifetime', in years

        product_weight:
        - Find in the assumptions table, usually labeled 'Product Weight', in kg

        ===== STEP 2: CALCULATE PERCENTAGES USING THE SENSITIVITY TABLE (PRIMARY METHOD) =====

        Look for a table on page 2 (sometimes labeled 'Sensitivity Analysis on Use Location of Product').
        It will have columns for each lifecycle stage and rows for Europe, China, and USA.
        The column headers may say: 'Manufacturing', 'Distribution' or 'Transportation', 'Use', 'End-of-life'.

        Use the EUROPE row values to calculate each percentage:
        - manufacturing % = round((Manufacturing_Europe / total_carbon_footprint) * 100, 1)
        - transportation % = round((Distribution_Europe / total_carbon_footprint) * 100, 1)
        - use % = round((Use_Europe / total_carbon_footprint) * 100, 1)
        - eol % = round((EndOfLife_Europe / total_carbon_footprint) * 100, 1)

        Note: the sum of the Europe row values should approximately equal the total carbon footprint.
        If they do not match closely, double-check that you have read the correct row and column values.

        ===== STEP 3: VERIFY USING THE PIE CHART (SECONDARY METHOD) =====

        Now look at the pie chart on page 1. It will show labeled percentage values on or near each segment.
        The legend will identify segments by number: 1=Manufacturing, 2=Transportation, 3=Use, 4=End-of-life.

        Read each labeled percentage directly off the chart and record them:
        - pie_manufacturing = the percentage on or near the largest segment
        - pie_transportation = the percentage on or near the second segment
        - pie_use = the percentage on or near the third segment
        - pie_eol = the percentage on or near the smallest segment (typically under 5%)

        ===== STEP 4: CROSS-REFERENCE AND RESOLVE =====

        Compare the table-derived values (Step 2) with the pie chart values (Step 3).
        They should be within ~1 percentage point of each other.

        - If they match closely (within 1%): use the pie-chart values as they are more precise.
        - If they disagree by more than 1%: re-examine both sources carefully.
        * Re-read the table row for Europe — ensure you have not confused rows or columns.
        * Re-examine the pie chart — ensure segment colors match the legend correctly.
        * Manufacturing is usually the largest value (typically 50-90%).
        * EoL is usually the smallest value (typically under 5%).
        * Transportation is typically 1-10%.
        * Use is typically 5-45% depending on the product type (desktops use more than laptops).
        * After re-examination, use whichever value you have higher confidence in and note the discrepancy.

        - Final sanity check: all four percentages should sum to approximately 100%.
        If they do not, something has been misread — re-examine before returning values.

        ===== STEP 5: RETURN RESULTS =====

        Return ONLY the JSON object with the final validated values. No explanation, no markdown fences.
        If a field cannot be found after thorough examination, use null."""
            }]

            #call openai client
            message = client.chat.completions.create(
                max_completion_tokens=10000, #max tokens
                messages=[{"role": "user", "content": message_content}], #give the prompt to the client
                model=deployment
            )

            pairs = {} #empty dictionary for metrics and their values
            model_name = "unknown" #initiate model name

            #extract the text from the message
            response_text = message.choices[0].message.content.strip()
            #logic to handle return structure
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            #load the data from json
            data = json.loads(response_text)

            #extract the model name
            model_name = data.get("model_name", "unknown")
            
            #extract carbon footprint information
            cf = data.get("carbon_footprint")
            #assign metric to dictionary
            pairs["This product's estimated carbon footprint:"] = f"{cf} kgCO2e" if cf is not None else None

            #extract manufacturing information
            mfg = data.get("manufacturing")
            #assign metric to dictionary
            pairs["Manufacturing"] = f"{mfg}%" if mfg is not None else None

            #extract transportation information
            trans = data.get("transportation")
            #assign metric to dictionary
            pairs["Transportation"] = f"{trans}%" if trans is not None else None

            #extract use information
            use = data.get("use")
            #assign metric to dictionary
            pairs["Use"] = f"{use}%" if use is not None else None

            #extract eol information
            eol = data.get("eol")
            #assign metric to dictionary
            pairs["EoL"] = f"{eol}%" if eol is not None else None

            #extract product lifetime information
            lt = data.get("product_lifetime")
            #assign metric to dictionary
            pairs["Product Lifetime"] = f"{lt} years" if lt is not None else None

            #extract product weight information
            pw = data.get("product_weight")
            #assign metric to dictionary
            pairs["Product Weight"] = f"{pw} kg" if pw is not None else None

            #assing model name
            pairs["model_name"] = model_name
            #assign pdf link
            pairs["link"] = pdf_link

            #assign brand name
            pairs["brand"] = "Dell"

            #assign category
            cat = re.search(r'/products/([^/]+)/', pdf_link).group(1).title()
            if cat == "Desktops-And-All-In-Ones":
                if model_name.__contains__("All-in-One"):
                    cat = "All in One System"
                else:
                    cat = "PC"
            elif cat == "Laptops-And-2-In-1S":
                cat = "Notebook"
            elif cat == "Electronics-And-Accessories":
                cat = "Monitor TFT"
            
            pairs["product_category"] = cat

            #assign pairs to full dictionary
            full_dict[model_name] = pairs

        except requests.exceptions.RequestException as e: #error catch for failed pdf download
            failures.append({"pdf": pdf_link, "error": f"Download failed: {str(e)}"})

        except json.JSONDecodeError as e: #error catch for invalid LLM output
            failures.append({"pdf": pdf_link, "error": f"LLM returned invalid JSON: {str(e)}"})

        except Exception as e: #any other errors
            failures.append({"pdf": pdf_link, "error": f"Unknown error: {str(e)}"})


    return full_dict



def get_dell_data(pairs_dict):

    data_list = [] #empty list for data

    for pairs in pairs_dict.values(): #loop through dictionary values

        
        cf_raw = pairs.get("This product's estimated carbon footprint:") #assign pcf info
        kg_CO2 = float(cf_raw.split("kgCO2e")[0]) if cf_raw is not None else None #return as a float


        mfg_raw = pairs.get("Manufacturing") #assign manufacturing info
        manufacturing_perc = float(mfg_raw.split("%")[0]) / 100 if mfg_raw is not None else None #return as a float and divide by 100
        manufacturing = kg_CO2 * manufacturing_perc if kg_CO2 is not None and manufacturing_perc is not None else None #compute manufacturing emissions


        trans_raw = pairs.get("Transportation") #assign transportation info
        transportation_perc = float(trans_raw.split("%")[0]) / 100 if trans_raw is not None else None #return as a float and divide by 100
        transport = kg_CO2 * transportation_perc if kg_CO2 is not None and transportation_perc is not None else None #compute transportation emissions


        use_raw = pairs.get("Use") #assign use info
        use_perc = float(use_raw.split("%")[0]) / 100 if use_raw is not None else None #return as a float and divide by 100
        usage = kg_CO2 * use_perc if kg_CO2 is not None and use_perc is not None else None #compute use emissions


        eol_raw = pairs.get("EoL") #assign eol info
        eol_perc = float(eol_raw.split("%")[0]) / 100 if eol_raw is not None else None #return as a float and divide by 100
        eol_em = kg_CO2 * eol_perc if kg_CO2 is not None and eol_perc is not None else None #compute eol emissions


        lt_raw = pairs.get("Product Lifetime") #assign lifetime info
        expected_lifespan = float(lt_raw.split("years")[0]) if lt_raw is not None else None #return as float
        usage_per_year = usage / expected_lifespan if usage is not None and expected_lifespan is not None else None #compute use/year


        pw_raw = pairs.get("Product Weight") #assign weight info
        prod_weight_kg = float(pw_raw.split("kg")[0]) if pw_raw is not None else None #return as float


        model_name = pairs.get("model_name", "unknown") #assign model name
        brand = pairs.get("brand", None) #assign brand
        category = pairs.get("product_category", None) #assign category
        pdf_link = pairs.get("link", None) #assign pdf link


        # ----- flags for if metrics were found ----- #
        if kg_CO2 == 0 or kg_CO2 is None:
            kg_CO2_flag = "No"
        else:
            kg_CO2_flag = "Yes"

        if manufacturing_perc == 0 or manufacturing_perc is None:
            manufacturing_flag = "No"
        else:
            manufacturing_flag = "Yes"

        if transportation_perc == 0 or transportation_perc is None:
            transport_flag = "No"
        else:
            transport_flag = "Yes"

        if use_perc == 0 or use_perc is None:
            use_flag = "No"
        else:
            use_flag = "Yes"

        if eol_perc == 0 or eol_perc is None:
            eol_flag = "No"
        else:
            eol_flag = "Yes"

        if expected_lifespan == 0 or expected_lifespan is None:
            lifespan_flag = "No"
        else:
            lifespan_flag = "Yes"

        if prod_weight_kg == 0 or prod_weight_kg is None:
            weight_flag = "No"
        else:
            weight_flag = "Yes"


        #storing model info as dictionary
        data = {
            "Product_Name": model_name,
            "Brand": brand,
            "Category": category,
            "Total_kgCO2e": kg_CO2,
            "Manufacturing_Perc": manufacturing_perc,
            "Manufacturing_Emission": manufacturing,
            "Transportation_Perc": transportation_perc,
            "Transportation_Emissions": transport,
            "Use_Perc": use_perc,
            "Use_Emissions": usage,
            "EOL_Perc": eol_perc,
            "EOL_Emissions": eol_em,
            "Expected_Lifespan_Years": expected_lifespan,
            "Use_per_Year": usage_per_year,
            "Product_Weight_kg": prod_weight_kg,
            "Carbon_Footprint_Flag": kg_CO2_flag,
            "Manufacturing_Flag": manufacturing_flag,
            "Transportation_Flag": transport_flag,
            "Use_Flag": use_flag,
            "EOL_Flag": eol_flag,
            "Lifespan_Flag": lifespan_flag,
            "Weight_Flag": weight_flag,
            "PDF_Link": pdf_link
        }

        #adding to data list
        data_list.append(data)

    df = pd.DataFrame(data_list) #converting to dataframe

    return df




#full function
def dell_func(old_pdfs):
    soup = get_dell_soup() #return html
    pairs = get_text_from_dell_pdf(soup, old_pdfs) #extract metric
    df = get_dell_data(pairs) #convert to dataframe
    return df









# -------- SAMSUNG -------- #

#function to pull html from samsung environment data website
def get_samsung_soup():
    url = "https://www.samsung.com/latin_en/sustainability/environment/environment-data/" #samsung url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
    } #headers to bypass blocking

    req = requests.get(url, headers=headers) #using requests to grab html

    req_h = req.text #extracting the text

    soup = BeautifulSoup(req_h, "html.parser") #parsing the html

    return soup #returns parsed html



#function for extracting information from pdfs
def get_text_from_samsung_pdf(soup):

    full_dict = {} #empty dictionary to store all information
    pdfs = [] #empty list to store pdf links


    for link in soup.find_all("a", href=re.compile(r'\.pdf')): #within the html, find all links for pdfs
        lab = link.get("aria-label", "").lower() #get the aria-label attribute of the link
        if lab.startswith("download lca_results_for_"): #filter for samsung lca pdfs only
            pdfs.append(link["href"]) # add pdf link to list

    pdfs = ["https:" + pdf_link for pdf_link in pdfs if pdf_link.startswith("//")]#prepend https to make a valid url


    failures = [] #empty list for storing failures

    
    for pdf_link in pdfs: #loop through all pdfs
        try:
            pdf = requests.get(pdf_link) #request pdf link
            pdf.raise_for_status() #raise error if download failed
            pdf_bytes = pdf.content #convert pdf to bytes

            # Use PyMuPDF to open the pdf from memory
            doc = fitz.open(stream=pdf_bytes, filename='pdf')

            first_product_bg = 0
            for i in range(min(8, len(doc))):
                if "Life Cycle Carbon Emissions" in doc[i].get_text():
                    first_product_bg = i - 1

                    break

            num_of_products = (len(doc) - first_product_bg) // 2

            for prod_i in tqdm(range(num_of_products), "Fetching Pages Within Samsung PDFs"): #loop through each product (2 pages each)
                
                image_contents = [] #list to store image content



                for page_num in [first_product_bg + prod_i * 2,
                                first_product_bg + prod_i * 2 + 1]: 

                    page = doc[page_num] #grab current page stored in doc
                    pix = page.get_pixmap(dpi=150) #render the page at 150 DPI
                    img_bytes = pix.tobytes('png') #convert the rendered page to png bytes
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8') #encode as base64 string

                    # Add each page as an image block in the format expected by the OpenAI vision API
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    })

            
                try:
                    #create prompt including image content and directions
                    message_content = image_contents + [{
                    "type": "text",
                    "text": """You are extracting structured data from Samsung Life Cycle Assessment (LCA) PDF pages.

                Some pages may be a Table of Contents or product index — skip those entirely. Only extract from pages that contain a "Life Cycle Carbon Emissions" bar chart and a "Product Features" table.

                Return ONLY a JSON array, one object per product page found. No explanation, no markdown fences, no extra text:

                [{"model_name":null,"carbon_footprint":null,"manufacturing":null,"transportation":null,"use":null,"eol":null,"product_lifetime":null,"product_weight":null}]

                If only one product page is found, still return a list with one object.
                If no valid product pages are found, return an empty list: []

                RULES:

                model_name:
                - Use the full product name from the page title e.g. "Galaxy A17 5G (US)".
                - Also acceptable: the Model name field in Product Features table (e.g. SM-A176U).

                carbon_footprint:
                - Find the "Life Cycle Carbon Emissions" horizontal bar chart (right side of page).
                - Totalling all of the numbers in the horizontal bar chart should yield the carbon footprint. You can double check this number in the chart titled "Characterized Environment Impact", under the Climate Change bar on the far left of the chart (e.g. 34.8).
                - Unit is kgCO₂ eq. Return the number only.
                - Do NOT use individual stage values separately.

                manufacturing:
                - From the bar chart on the right hand side, find the Manufacturing segment value.
                - Divide by total carbon_footprint and multiply by 100. Return to 1 decimal place.

                transportation:
                - From the bar chart chart on the right hand side, find the Distribution segment value. (Distribution = Transportation)
                - Divide by total and multiply by 100. Return to 1 decimal place.

                use:
                - From the bar chart chart on the right hand side, find the Use segment value.
                - Divide by total and multiply by 100. Return to 1 decimal place.

                eol:
                - From the bar chart chart on the right hand side, find the Disposal segment value. (Disposal = End of Life)
                - Divide by total and multiply by 100. Return to 1 decimal place.
                - If 0 or not visible, return 0.

                product_lifetime:
                - Find the label under the bar chart that says "Emission for X years".
                - Return only the number X.

                product_weight:
                - Find the Product Features table, Weight row.
                - Use the value next to "Product & Acc." (NOT Packages).
                - Value is in grams — divide by 1000 to convert to kg.
                - Return in kg to 2 decimal places (e.g. 221.15g → 0.22).

                VALIDATION (internal only, do not output):
                - manufacturing + transportation + use + eol must equal 100.
                - If not, recheck — you likely used the wrong total or missed a segment.

                Every field must be present in each object. Use null only if truly not found."""
                }]

                    #call openai client
                    message = client.chat.completions.create(
                        max_completion_tokens=10000, #max tokens
                        messages=[{"role": "user", "content": message_content}], #give the prompt to the client
                        model=deployment
                    )

                    model_name = "unknown" #initiate model name

                    #extract the text from the message
                    response_text = message.choices[0].message.content.strip()
                    #logic to handle return structure
                    if response_text.startswith('```'):
                        response_text = response_text.split('```')[1]
                        if response_text.startswith('json'):
                            response_text = response_text[4:]

                    #load the data from json — returns a list since samsung pdfs can contain multiple products
                    data = json.loads(response_text)

                    for prod in data: #loop through each product found in the pdf
                        pairs = {} #empty dictionary for metrics and their values

                        #extract model name
                        model_name = prod.get("model_name", "unknown")

                        #extract carbon footprint information
                        cf = prod.get("carbon_footprint")
                        #assign metric to dictionary
                        pairs["This product's estimated carbon footprint:"] = f"{cf} kgCO2e" if cf is not None else None

                        #extract manufacturing information
                        mfg = prod.get("manufacturing")
                        #assign metric to dictionary
                        pairs["Manufacturing"] = f"{mfg}%" if mfg is not None else None

                        #extract transportation information
                        trans = prod.get("transportation")
                        #assign metric to dictionary
                        pairs["Transportation"] = f"{trans}%" if trans is not None else None

                        #extract use information
                        use = prod.get("use")
                        #assign metric to dictionary
                        pairs["Use"] = f"{use}%" if use is not None else None

                        #extract eol information
                        eol = prod.get("eol")
                        #assign metric to dictionary
                        pairs["EoL"] = f"{eol}%" if eol is not None else None

                        #extract product lifetime information
                        lt = prod.get("product_lifetime")
                        #assign metric to dictionary
                        pairs["Product Lifetime"] = f"{lt} years" if lt is not None else None

                        #extract product weight information
                        pw = prod.get("product_weight")
                        #assign metric to dictionary
                        pairs["Product Weight"] = f"{pw} kg" if pw is not None else None

                        #assign model name
                        pairs["model_name"] = model_name
                        #assign pdf link
                        pairs["link"] = pdf_link

                        #assign brand
                        pairs["brand"] = "Samsung"
                        
                        #assign category
                        cat = re.search(r"for%20([^%/]+)\.pdf$", pdf_link).group(1).title()
                        if cat == "Smartphones":
                            cat = "Mobile Telephone"
                        elif cat == "Tablets":
                            cat = "Tablet"

                        pairs["product_category"] = cat

                        #assign pairs to full dictionary using model name as key
                        full_dict[model_name] = pairs

                except json.JSONDecodeError as e:
                    failures.append({"pdf": pdf_link, "pages": [first_product_bg + prod_i * 2, first_product_bg + prod_i * 2 + 1], "error": f"Invalid JSON: {str(e)}"})
                except Exception as e:
                    failures.append({"pdf": pdf_link, "pages": [first_product_bg + prod_i * 2, first_product_bg + prod_i * 2 + 1], "error": f"API error: {str(e)}"})        

            doc.close() #free pdf from memory once all pages have been processed

            time.sleep(1)

        except requests.exceptions.RequestException as e: #error catch for failed pdf download
            failures.append({"pdf": pdf_link, "error": f"Download failed: {str(e)}"})

        except json.JSONDecodeError as e: #error catch for invalid LLM output
            failures.append({"pdf": pdf_link, "error": f"LLM returned invalid JSON: {str(e)}"})

        except Exception as e: #any other errors
            failures.append({"pdf": pdf_link, "error": f"Unknown error: {str(e)}"})

    return full_dict







def get_samsung_data(pairs_dict):

    data_list = [] #empty list for data

    for pairs in pairs_dict.values(): #loop through dictionary values

        cf_raw = pairs.get("This product's estimated carbon footprint:") #assign pcf info
        kg_CO2 = float(cf_raw.split("kgCO2e")[0]) if cf_raw is not None else None #return as a float

        mfg_raw = pairs.get("Manufacturing") #assign manufacturing info
        manufacturing_perc = float(mfg_raw.split("%")[0]) / 100 if mfg_raw is not None else None #return as a float and divide by 100
        manufacturing = kg_CO2 * manufacturing_perc if kg_CO2 is not None and manufacturing_perc is not None else None #compute manufacturing emissions

        trans_raw = pairs.get("Transportation") #assign transportation info
        transportation_perc = float(trans_raw.split("%")[0]) / 100 if trans_raw is not None else None #return as a float and divide by 100
        transport = kg_CO2 * transportation_perc if kg_CO2 is not None and transportation_perc is not None else None #compute transportation emissions

        use_raw = pairs.get("Use") #assign use info
        use_perc = float(use_raw.split("%")[0]) / 100 if use_raw is not None else None #return as a float and divide by 100
        usage = kg_CO2 * use_perc if kg_CO2 is not None and use_perc is not None else None #compute use emissions

        eol_raw = pairs.get("EoL") #assign eol info
        eol_perc = float(eol_raw.split("%")[0]) / 100 if eol_raw is not None else None #return as a float and divide by 100
        eol_em = kg_CO2 * eol_perc if kg_CO2 is not None and eol_perc is not None else None #compute eol emissions

        lt_raw = pairs.get("Product Lifetime") #assign lifetime info
        expected_lifespan = float(lt_raw.split("years")[0]) if lt_raw is not None else None #return as float
        usage_per_year = usage / expected_lifespan if usage is not None and expected_lifespan is not None else None #compute use/year

        pw_raw = pairs.get("Product Weight") #assign weight info
        prod_weight_kg = float(pw_raw.split("kg")[0]) if pw_raw is not None else None #return as float

        # ----- flags for if metrics were found ----- #
        if kg_CO2 == 0 or kg_CO2 is None:
            kg_CO2_flag = "No"
        else:
            kg_CO2_flag = "Yes"

        if manufacturing_perc == 0 or manufacturing_perc is None:
            manufacturing_flag = "No"
        else:
            manufacturing_flag = "Yes"

        if transportation_perc == 0 or transportation_perc is None:
            transport_flag = "No"
        else:
            transport_flag = "Yes"

        if use_perc == 0 or use_perc is None:
            use_flag = "No"
        else:
            use_flag = "Yes"

        if eol_perc == 0 or eol_perc is None:
            eol_flag = "No"
        else:
            eol_flag = "Yes"

        if expected_lifespan == 0 or expected_lifespan is None:
            lifespan_flag = "No"
        else:
            lifespan_flag = "Yes"

        if prod_weight_kg == 0 or prod_weight_kg is None:
            weight_flag = "No"
        else:
            weight_flag = "Yes"

        #storing model info as dictionary
        data = {
            "Product_Name": pairs["model_name"],
            "Brand": pairs["brand"],
            "Category": pairs["product_category"],
            "Total_kgCO2e": kg_CO2,
            "Manufacturing_Perc": manufacturing_perc,
            "Manufacturing_Emission": manufacturing,
            "Transportation_Perc": transportation_perc,
            "Transportation_Emissions": transport,
            "Use_Perc": use_perc,
            "Use_Emissions": usage,
            "EOL_Perc": eol_perc,
            "EOL_Emissions": eol_em,
            "Expected_Lifespan_Years": expected_lifespan,
            "Use_per_Year": usage_per_year,
            "Product_Weight_kg": prod_weight_kg,
            "Carbon_Footprint_Flag": kg_CO2_flag,
            "Manufacturing_Flag": manufacturing_flag,
            "Transportation_Flag": transport_flag,
            "Use_Flag": use_flag,
            "EOL_Flag": eol_flag,
            "Lifespan_Flag": lifespan_flag,
            "Weight_Flag": weight_flag,
            "PDF_Link": pairs["link"]
        }

        #adding to data list
        data_list.append(data)

    df = pd.DataFrame(data_list) #converting to dataframe

    return df



def samsung_func():
    soup = get_samsung_soup() #return html
    pairs = get_text_from_samsung_pdf(soup) #extract metrics
    df = get_samsung_data(pairs) #convert to dataframe
    return df








# ------------ Lenovo ------------ #
 


#function to pull javascript file containing lenovo pcf pdf links
def get_lenovo_soup():
    url = "https://j1-ofp.static.pub/ShareResource/esg/scripts/eco-declarations-main-2025.js" #lenovo eco declarations javascript url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
    } #headers to bypass blocking

    req = requests.get(url, headers=headers) #using requests to grab html

    req_h = req.text #extracting the text

    soup = BeautifulSoup(req_h, "html.parser") #parsing the html

    return soup #returns parsed html



#function for extracting information from pdfs
def get_text_from_lenovo_pdf(soup, old_pdfs=None):

    full_dict = {} #empty dictionary to store all information

    old_pdfs = set(old_pdfs) if old_pdfs is not None else set() #ensure old_pdfs is a set

    pattern = re.compile(f"PCF.*", re.I) #regex pattern to match links containing "PCF" (case insensitive)
    lap = soup.find_all("a", string=lambda text: text and pattern.search(text)) #find all anchor tags whose text matches the PCF pattern

    pdfs = ["https:" + l['href'] for l in lap if l['href'].startswith('//')] #prepend https to make a valid url
    
    pdfs = [pdf for pdf in pdfs if pdf not in old_pdfs] #list new pdfs not already present

    failures = []


    
    for link in tqdm(pdfs, "Fetching Lenovo PCF Information"): #loop through first 30 matching pdf links
        try:
            pdf = requests.get(link) #request pdf from url
            pdf.raise_for_status() #raise error if download failed
            pdf_bytes = pdf.content #convert pdf to bytes

            # Use PyMuPDF to open the pdf from memory
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            image_content = [] #list to store image content

            for page_num in range(len(doc)): #loop through each page of pdf
                page = doc[page_num] #grab current page stored in doc
                pix = page.get_pixmap(dpi=200) #render the page at 200 DPI for clarity
                img_bytes = pix.tobytes('png') #convert the rendered page to png bytes
                img_base64 = base64.b64encode(img_bytes).decode('utf-8') #encode as base64 string

                # Add each page as an image block in the format expected by the OpenAI vision API
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })

            doc.close() #free pdf from memory once all pages have been processed

            #create prompt including image content and directions
            message_content = image_content + [{
            "type": "text",
            "text": """
        You are extracting structured data from a Lenovo Product Carbon Footprint PDF.

        Return ONLY one valid JSON object with exactly these keys and no extra text:

        {
        "model_name": null,
        "carbon_footprint": null,
        "manufacturing": null,
        "transportation": null,
        "use": null,
        "eol": null,
        "product_lifetime": null,
        "product_weight": null,
        "product_category": null
        }

        Extraction rules:
        1. model_name:
        - Use the value next to "Commercial Name".
        - Do NOT use the model number unless the commercial name is missing.

        2. carbon_footprint:
        - Use the main Product Carbon Footprint Value shown on page 1.
        - Return the numeric value only, with no units.
        - If the PDF shows both a headline PCF value and a mean +/- std value in a note, use the headline PCF value, not the mean.

        3. manufacturing / transportation / use / eol:
        - These should be lifecycle percentages from the pie chart.
        - Return numbers only, no percent signs.
        - transportation = the slice labeled Transportation.
        - use = the slice labeled Use.
        - eol = the slice labeled EoL or End of Life.
        - manufacturing = sum of all remaining product/component slices in the pie chart that are NOT transportation, use, or eol.
        - Include slices labeled 0% as 0.
        - If a slice is present but shown as 0%, treat it as 0, not null.
        - If manufacturing is not explicitly labeled, compute it by summing component slices such as display, motherboard, battery, power supply, chassis, packaging, hard drive, optical drive, housing, panel, etc.

        - It is common for EoL to be 0%. Make sure that you include it as 0 if it is 0%.

        4. product_lifetime:
        - Use the value for "Product Lifetime" from the assumptions table.
        - Return the number only.

        5. product_weight:
        - Use the value for "Product Weight" from the assumptions table.
        - Return the number only.

        6. product_category:
        - Use information to extract the product type to the best of your ability (eg. Tablet, PC, Notebook, Monitor TFT)
        - A good indicator is the picture of the product.

        7. General rules:
        - The needed fields may be spread across multiple pages.
        - Read text, tables, and chart labels carefully.
        - Preserve decimals exactly when present.
        - Use null only if a field cannot be found.
        - Output must be valid JSON only.

        8. Validation:
        - manufacturing + transportation + use + eol should approximately equal 100.
        - Do not output the validation math, just use it internally to avoid mistakes.
        """
        }]

            #call openai client
            message = client.chat.completions.create(
                max_completion_tokens=1024, #max tokens
                messages=[{"role": "user", "content": message_content}], #give the prompt to the client
                model=deployment
            )

            pairs = {} #empty dictionary for metrics and their values
            model_name = "unknown" #initiate model name

            #extract the text from the message
            response_text = message.choices[0].message.content.strip()
            #logic to handle return structure
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            #load the data from json
            data = json.loads(response_text)

            #extract the model name
            model_name = data.get("model_name", "unknown")

            #extract carbon footprint information
            cf = data.get("carbon_footprint")
            #assign metric to dictionary
            pairs["This product's estimated carbon footprint:"] = f"{cf} kgCO2e" if cf is not None else None

            #extract manufacturing information
            mfg = data.get("manufacturing")
            #assign metric to dictionary
            pairs["Manufacturing"] = f"{mfg}%" if mfg is not None else None

            #extract transportation information
            trans = data.get("transportation")
            #assign metric to dictionary
            pairs["Transportation"] = f"{trans}%" if trans is not None else None

            #extract use information
            use = data.get("use")
            #assign metric to dictionary
            pairs["Use"] = f"{use}%" if use is not None else None

            #extract eol information
            eol = data.get("eol")
            #assign metric to dictionary
            pairs["EoL"] = f"{eol}%" if eol is not None else None

            #extract product lifetime information
            lt = data.get("product_lifetime")
            #assign metric to dictionary
            pairs["Product Lifetime"] = f"{lt} years" if lt is not None else None

            #extract product weight information
            pw = data.get("product_weight")
            #assign metric to dictionary
            pairs["Product Weight"] = f"{pw} kg" if pw is not None else None

            #assign model name
            pairs["model_name"] = model_name

            #assign product category
            category = data.get("product_category")
            pairs["category"] = category if category is not None else None

            #assign pdf link
            pairs["link"] = link




            #assign pairs to full dictionary using model name as key
            full_dict[model_name] = pairs

        except requests.exceptions.RequestException as e: #error catch for failed pdf download
            failures.append({"pdf": link, "error": f"Download failed: {str(e)}"})

        except json.JSONDecodeError as e: #error catch for invalid LLM output
            failures.append({"pdf": link, "error": f"LLM returned invalid JSON: {str(e)}"})

        except Exception as e: #any other errors
            failures.append({"pdf": link, "error": f"Unknown error: {str(e)}"})


    return full_dict #return full dictionary



def get_lenovo_data(pairs_dict):

    data_list = [] #empty list for data

    for pairs in pairs_dict.values(): #loop through dictionary values

        cf_raw = pairs.get("This product's estimated carbon footprint:") #assign pcf info
        kg_CO2 = float(cf_raw.split("kgCO2e")[0]) if cf_raw is not None else None #return as a float

        mfg_raw = pairs.get("Manufacturing") #assign manufacturing info
        manufacturing_perc = float(mfg_raw.split("%")[0]) / 100 if mfg_raw is not None else None #return as a float and divide by 100
        manufacturing = kg_CO2 * manufacturing_perc if kg_CO2 is not None and manufacturing_perc is not None else None #compute manufacturing emissions

        trans_raw = pairs.get("Transportation") #assign transportation info
        transportation_perc = float(trans_raw.split("%")[0]) / 100 if trans_raw is not None else None #return as a float and divide by 100
        transport = kg_CO2 * transportation_perc if kg_CO2 is not None and transportation_perc is not None else None #compute transportation emissions

        use_raw = pairs.get("Use") #assign use info
        use_perc = float(use_raw.split("%")[0]) / 100 if use_raw is not None else None #return as a float and divide by 100
        usage = kg_CO2 * use_perc if kg_CO2 is not None and use_perc is not None else None #compute use emissions

        eol_raw = pairs.get("EoL") #assign eol info
        eol_perc = float(eol_raw.split("%")[0]) / 100 if eol_raw is not None else None #return as a float and divide by 100
        eol_em = kg_CO2 * eol_perc if kg_CO2 is not None and eol_perc is not None else None #compute eol emissions

        lt_raw = pairs.get("Product Lifetime") #assign lifetime info
        expected_lifespan = float(lt_raw.split("years")[0]) if lt_raw is not None else None #return as float
        usage_per_year = usage / expected_lifespan if usage is not None and expected_lifespan is not None else None #compute use/year

        pw_raw = pairs.get("Product Weight") #assign weight info
        prod_weight_kg = float(pw_raw.split("kg")[0]) if pw_raw is not None else None #return as float

        # ----- flags for if metrics were found ----- #
        if kg_CO2 == 0 or kg_CO2 is None:
            kg_CO2_flag = "No"
        else:
            kg_CO2_flag = "Yes"

        if manufacturing_perc == 0 or manufacturing_perc is None:
            manufacturing_flag = "No"
        else:
            manufacturing_flag = "Yes"

        if transportation_perc == 0 or transportation_perc is None:
            transport_flag = "No"
        else:
            transport_flag = "Yes"

        if use_perc == 0 or use_perc is None:
            use_flag = "No"
        else:
            use_flag = "Yes"

        if eol_perc == 0 or eol_perc is None:
            eol_flag = "No"
        else:
            eol_flag = "Yes"

        if expected_lifespan == 0 or expected_lifespan is None:
            lifespan_flag = "No"
        else:
            lifespan_flag = "Yes"

        if prod_weight_kg == 0 or prod_weight_kg is None:
            weight_flag = "No"
        else:
            weight_flag = "Yes"

        #storing model info as dictionary
        data = {
            "Product_Name": pairs["model_name"],
            "Brand": "Lenovo",
            "Category": pairs["category"],
            "Total_kgCO2e": kg_CO2,
            "Manufacturing_Perc": manufacturing_perc,
            "Manufacturing_Emission": manufacturing,
            "Transportation_Perc": transportation_perc,
            "Transportation_Emissions": transport,
            "Use_Perc": use_perc,
            "Use_Emissions": usage,
            "EOL_Perc": eol_perc,
            "EOL_Emissions": eol_em,
            "Expected_Lifespan_Years": expected_lifespan,
            "Use_per_Year": usage_per_year,
            "Product_Weight_kg": prod_weight_kg,
            "Carbon_Footprint_Flag": kg_CO2_flag,
            "Manufacturing_Flag": manufacturing_flag,
            "Transportation_Flag": transport_flag,
            "Use_Flag": use_flag,
            "EOL_Flag": eol_flag,
            "Lifespan_Flag": lifespan_flag,
            "Weight_Flag": weight_flag,
            "PDF_Link": pairs["link"]
        }

        #adding to data list
        data_list.append(data)

    df = pd.DataFrame(data_list) #converting to dataframe

    return df





#full function
def lenovo_func(old_pdfs):
    soup = get_lenovo_soup() #return html
    pairs = get_text_from_lenovo_pdf(soup, old_pdfs) #extract metrics
    df = get_lenovo_data(pairs) #convert to dataframe
    return df








# --------------- Apple --------------- #



def get_apple_soup():
    url = "https://www.apple.com/environment/" #apple url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
    } #headers to bypass blocking

    req = requests.get(url, headers=headers) #using requests to grab html

    req_h = req.text #extracting the text

    soup = BeautifulSoup(req_h, "html.parser") #parsing the html

    return soup #returns parsed html






#function for extracting information from pdfs
def get_text_from_apple_pdf(soup, old_pdfs=None):

    full_dict = {} #empty dictionary to store all information
    links = [] #empty list to store pdf links

    old_pdfs = set(old_pdfs) if old_pdfs is not None else set() #ensure old_pdfs is a set

    for link in soup.find_all("a", href=re.compile(r'\.pdf')): #look through each pdf link in the html
        analytics_title = link.get('data-analytics-title', '').lower() #get the title of the pdf, stored under data-analytics-title

        clean_title = analytics_title.replace(' - view pdf', '').strip() #remove "view pdf" suffix from title

        pdf_link = "https://www.apple.com" + link['href'] #prepend apple base url to create full pdf link

        links.append(pdf_link) # add pdf link to list

    pdfs = [pdf for pdf in links if pdf not in old_pdfs]

    failures = []

    
    for pdf_link in tqdm(pdfs, "Fetching Apple PCF Information"): #loop through first 3 pdf links
        try:
            pdf = requests.get(pdf_link) #request pdf from url
            pdf.raise_for_status() #raise error if download failed
            pdf_bytes = pdf.content #convert pdf to bytes

            # Use PyMuPDF to open the pdf from memory
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            image_content = [] #list to store image content

            pcf_page = None #initiate pcf page tracker as None
            for page_num, page in enumerate(doc): #loop through each page to find the carbon footprint page
                text = page.get_text().lower() #extract text from page and convert to lowercase
                if ('production' in text or 'manufacturing' in text) and 'kg' in text and '%' in text: #check if page contains carbon footprint data
                    pcf_page = page_num #store the page number where pcf data was found
                    break #stop searching once the relevant page is found

            if pcf_page is None: #if no carbon footprint page was found
                doc.close() #free pdf from memory
                continue #skip to the next pdf

            for page_num in range(len(doc)): #loop through all pages of the pdf
                page = doc[page_num] #grab current page stored in doc
                pix = page.get_pixmap(dpi=200) #render the page at 200 DPI for clarity
                img_bytes = pix.tobytes('png') #convert the rendered page to png bytes
                img_base64 = base64.b64encode(img_bytes).decode('utf-8') #encode as base64 string

                # Add each page as an image block in the format expected by the OpenAI vision API
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })

            doc.close() #free pdf from memory once all pages have been processed

            #create prompt including image content and directions
            message_content = image_content + [{
            "type": "text",
            "text": """You are extracting structured carbon footprint data from Apple Product Environmental Reports (PERs).

            Return ONLY a JSON array — one object per storage/configuration variant found. No explanation, no markdown fences.

            Each object must have these exact keys:
            {
            "model_name": null,
            "configuration": null,
            "carbon_footprint": null,
            "manufacturing": null,
            "transportation": null,
            "use": null,
            "eol": null,
            "product_lifetime": null,
            "product_weight": null,
            "product_category": null
            }

            ===== STEP 1: EXTRACT THE MODEL NAME =====

            model_name:
            - Some Apple PERs cover multiple products in a single document (e.g., "iPhone 17 Pro and iPhone 17 Pro Max").
            - Do NOT use the document title as the model_name if it combines multiple products with "and".
            - Instead, treat each product as completely separate with its own distinct model_name.
            - Extract the individual product name from context — the configuration table, chart labels, 
            or section headings will identify each product separately.
            e.g., document title "iPhone 17 Pro and iPhone 17 Pro Max PER" → 
            model_name values should be "iPhone 17 Pro" and "iPhone 17 Pro Max" separately.
            - Never include "and [other product]" in a model_name field.

            ===== STEP 2: EXTRACT LIFECYCLE PERCENTAGES =====

            Apple reports percentages in one of three formats. Check all three:

            FORMAT A — Plain text list (e.g., iPhone SE style):
            Look for a section titled "[Product] life cycle carbon emissions" with a bulleted or inline list:
            e.g., "82% Production  4% Transport  13% Use  <1% End-of-life processing"
            Use these values directly.

            FORMAT B — Table (e.g., iMac style):
            Look for a table near the end of the document (often on a "Carbon Footprint" data page) with rows like:
            "Production 50%  Transportation 5%  Product use 45%  End-of-life processing <1%"
            Use these values directly.

            FORMAT C — Donut/pie chart with multiple production sub-segments (e.g., iPhone 17 style):
            The chart splits production into sub-categories such as:
                - "Production: Materials and Process Emissions" (e.g., 53%)
                - "Production: Electricity" (e.g., 23%)
                - "Renewable Energy Emissions" (a negative credit, e.g., 1%)
                - "Electricity for Charging" (this is the USE stage, e.g., 18%)
                - "Transportation" (e.g., 4%)
                - "End-of-life Processing" (e.g., <1%)
            
            Combine segments as follows:
                manufacturing = (Production: Materials and Process) + (Production: Electricity) - (Renewable Energy Emissions)
                use = Electricity for Charging
                transportation = Transportation
                eol = End-of-life Processing

            FIELD DEFINITIONS:
            - manufacturing: All production/manufacturing phase emissions as a percentage of total. See FORMAT C above for combining sub-segments.
            - transportation: Transport/distribution/shipping phase only.
            - use: Product use phase (electricity for charging, power consumption over lifetime).
            - eol: End-of-life processing. If shown as "<1%", store as 0.5.
            - All percentages should sum to approximately 100. If they don't, re-examine the source.

            ===== STEP 3: EXTRACT PRODUCT LIFETIME =====

            product_lifetime:
            - Look in the Definitions section for language like:
            "Apple assumes a three-year period for [device type]" or "four-year period for [device type]"
            - iPhones, Apple Watch, AirPods → 3 years
            - Mac, iPad, Apple TV, Vision Pro → 4 years
            - Return the number only (e.g., 3 or 4)

            ===== STEP 4: PRODUCT WEIGHT and CATEGORY =====

            product_weight:
            - Apple Product Environmental Reports do NOT include product weight.
            - Always return null for this field.

            product category:
            - Use your best judjement to find the category of the product
            - Choose from this list (Tablet, Mobile Telephone, All in One System, PC, Notebook)

            ===== STEP 5: EXTRACT ALL CONFIGURATIONS =====

            Near the end of the document, look for a table listing carbon footprint values. 
            This table may contain MULTIPLE distinct products AND multiple configurations per product.
            For example:

            Configuration     iPhone 17 Pro    iPhone 17 Pro Max
            256GB             85 kg CO2e       95 kg CO2e
            512GB             91 kg CO2e       101 kg CO2e
            1TB               103 kg CO2e      113 kg CO2e

            Treat each UNIQUE combination of product model + storage as a separate JSON object.
            - "model_name" must reflect the specific product (e.g., 'iPhone 17 Pro' vs 'iPhone 17 Pro Max')
            - "configuration" is the storage size (e.g., '256GB')
            - Each product variant may have different lifecycle percentages — extract them separately
            - Do NOT share percentages between different product models unless they are explicitly identical

            ===== STEP 6: HANDLE EDGE CASES =====

            - "<1%" values: store as 0.5
            - If a percentage is not found after thorough examination, use null
            - Percentages should be stored as numbers only, not strings (e.g., 82.0 not "82%")
            - carbon_footprint should be a number only (e.g., 46 not "46 kg CO2e")
            - If multiple products appear in one document, return objects for each
            - "configuration" must ONLY contain the storage size (e.g., "256GB", "512GB", "1TB"). Never include the product name or any other text in the configuration field.

            ===== FINAL VALIDATION =====

            Before returning:
            - manufacturing + transportation + use + eol should sum to approximately 100 (±2 to account for rounding and <1% entries)
            - Each configuration should have its own carbon_footprint value
            - model_name should be identical across all objects from the same document

            Return ONLY the JSON array. No explanation, no markdown."""
        }]

            #call openai client
            message = client.chat.completions.create(
                max_completion_tokens=10000, #max tokens
                messages=[{"role": "user", "content": message_content}], #give the prompt to the client
                model=deployment
            )

            pairs = {} #empty dictionary for metrics and their values
            model_name = "unknown" #initiate model name

            #extract the text from the message
            response_text = message.choices[0].message.content.strip()
            #logic to handle return structure
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            #load the data from json — returns a list since apple PERs can contain multiple products and configurations
            data = json.loads(response_text)

            for prod in data: #loop through each product/configuration combination found in the pdf
                pairs = {} #empty dictionary for metrics and their values

                #extract model name
                model_name = prod.get("model_name", "unknown")
                product_category = prod.get("product_category", None)
                config = prod.get("configuration") #extract configuration e.g. storage size in GB

                #extract carbon footprint information
                cf = prod.get("carbon_footprint")
                #assign metric to dictionary
                pairs["This product's estimated carbon footprint:"] = f"{cf} kgCO2e" if cf is not None else None

                #extract manufacturing information
                mfg = prod.get("manufacturing")
                #assign metric to dictionary
                pairs["Manufacturing"] = f"{mfg}%" if mfg is not None else None

                #extract transportation information
                trans = prod.get("transportation")
                #assign metric to dictionary
                pairs["Transportation"] = f"{trans}%" if trans is not None else None

                #extract use information
                use = prod.get("use")
                #assign metric to dictionary
                pairs["Use"] = f"{use}%" if use is not None else None

                #extract eol information
                eol = prod.get("eol")
                #assign metric to dictionary
                pairs["EoL"] = f"{eol}%" if eol is not None else None

                #extract product lifetime information
                lt = prod.get("product_lifetime")
                #assign metric to dictionary
                pairs["Product Lifetime"] = f"{lt} years" if lt is not None else None

                #extract product weight information (always null for apple)
                pw = prod.get("product_weight")
                #assign metric to dictionary
                pairs["Product Weight"] = f"{pw} kg" if pw is not None else None

                pairs["Product Category"] = product_category

                #assign model name
                pairs["model_name"] = model_name
                #assign pdf link
                pairs["link"] = pdf_link

                #build unique key from model name and configuration to avoid overwriting products with the same name
                dict_key = f"{model_name} {config}" if config else model_name
                #assign pairs to full dictionary using the unique key
                full_dict[dict_key] = pairs

        except requests.exceptions.RequestException as e: #error catch for failed pdf download
            failures.append({"pdf": pdf_link, "error": f"Download failed: {str(e)}"})

        except json.JSONDecodeError as e: #error catch for invalid LLM output
            failures.append({"pdf": pdf_link, "error": f"LLM returned invalid JSON: {str(e)}"})

        except Exception as e: #any other errors
            failures.append({"pdf": pdf_link, "error": f"Unknown error: {str(e)}"})

    return full_dict #return full dictionary of all products and their metrics




def get_apple_data(pairs_dict):

    data_list = [] #empty list for data

    for key, pairs in pairs_dict.items(): #loop through dictionary keys and values

        cf_raw = pairs.get("This product's estimated carbon footprint:") #assign pcf info
        kg_CO2 = float(cf_raw.split("kgCO2e")[0]) if cf_raw is not None else None #return as a float

        mfg_raw = pairs.get("Manufacturing") #assign manufacturing info
        manufacturing_perc = float(mfg_raw.split("%")[0]) / 100 if mfg_raw is not None else None #return as a float and divide by 100
        manufacturing = kg_CO2 * manufacturing_perc if kg_CO2 is not None and manufacturing_perc is not None else None #compute manufacturing emissions

        trans_raw = pairs.get("Transportation") #assign transportation info
        transportation_perc = float(trans_raw.split("%")[0]) / 100 if trans_raw is not None else None #return as a float and divide by 100
        transport = kg_CO2 * transportation_perc if kg_CO2 is not None and transportation_perc is not None else None #compute transportation emissions

        use_raw = pairs.get("Use") #assign use info
        use_perc = float(use_raw.split("%")[0]) / 100 if use_raw is not None else None #return as a float and divide by 100
        usage = kg_CO2 * use_perc if kg_CO2 is not None and use_perc is not None else None #compute use emissions

        eol_raw = pairs.get("EoL") #assign eol info
        eol_perc = float(eol_raw.split("%")[0]) / 100 if eol_raw is not None else None #return as a float and divide by 100
        eol_em = kg_CO2 * eol_perc if kg_CO2 is not None and eol_perc is not None else None #compute eol emissions

        lt_raw = pairs.get("Product Lifetime") #assign lifetime info
        expected_lifespan = float(lt_raw.split("years")[0]) if lt_raw is not None else None #return as float
        usage_per_year = usage / expected_lifespan if usage is not None and expected_lifespan is not None else None #compute use/year

        pw_raw = pairs.get("Product Weight") #assign weight info
        prod_weight_kg = float(pw_raw.split("kg")[0]) if pw_raw is not None else None #return as float

        category = pairs.get("Product Category")

        # ----- flags for if metrics were found ----- #
        if kg_CO2 == 0 or kg_CO2 is None:
            kg_CO2_flag = "No"
        else:
            kg_CO2_flag = "Yes"

        if manufacturing_perc == 0 or manufacturing_perc is None:
            manufacturing_flag = "No"
        else:
            manufacturing_flag = "Yes"

        if transportation_perc == 0 or transportation_perc is None:
            transport_flag = "No"
        else:
            transport_flag = "Yes"

        if use_perc == 0 or use_perc is None:
            use_flag = "No"
        else:
            use_flag = "Yes"

        if eol_perc == 0 or eol_perc is None:
            eol_flag = "No"
        else:
            eol_flag = "Yes"

        if expected_lifespan == 0 or expected_lifespan is None:
            lifespan_flag = "No"
        else:
            lifespan_flag = "Yes"

        if prod_weight_kg == 0 or prod_weight_kg is None:
            weight_flag = "No"
        else:
            weight_flag = "Yes"

        #storing model info as dictionary — uses the full dict key (model + config) as Product_Name
        data = {
            "Product_Name": key,
            "Brand": "Apple",
            "Category": category,
            "Total_kgCO2e": kg_CO2,
            "Manufacturing_Perc": manufacturing_perc,
            "Manufacturing_Emission": manufacturing,
            "Transportation_Perc": transportation_perc,
            "Transportation_Emissions": transport,
            "Use_Perc": use_perc,
            "Use_Emissions": usage,
            "EOL_Perc": eol_perc,
            "EOL_Emissions": eol_em,
            "Expected_Lifespan_Years": expected_lifespan,
            "Use_per_Year": usage_per_year,
            "Product_Weight_kg": prod_weight_kg,
            "Carbon_Footprint_Flag": kg_CO2_flag,
            "Manufacturing_Flag": manufacturing_flag,
            "Transportation_Flag": transport_flag,
            "Use_Flag": use_flag,
            "EOL_Flag": eol_flag,
            "Lifespan_Flag": lifespan_flag,
            "Weight_Flag": weight_flag,
            "PDF_Link": pairs["link"]
        }

        #adding to data list
        data_list.append(data)

    df = pd.DataFrame(data_list) #converting to dataframe

    return df





#full function
def apple_func(old_pdfs):
    soup = get_apple_soup() #return html
    pairs = get_text_from_apple_pdf(soup, old_pdfs) #extract metrics
    df = get_apple_data(pairs) #convert to dataframe
    return df