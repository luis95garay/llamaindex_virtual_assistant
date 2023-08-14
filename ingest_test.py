# from langchain.document_loaders import BSHTMLLoader, UnstructuredHTMLLoader

# loader = UnstructuredHTMLLoader("https://learn.microsoft.com/en-us/sharepoint/dev/sp-add-ins/navigate-the-sharepoint-data-structure-represented-in-the-rest-service")
# data = loader.load()
# data


import requests

url = 'https://asistenciaglobal.sharepoint.com/sites/bluemedicalintranet/SitePages/Fisioterapia.aspx'
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
    print(html_content)
else:
    print(f"Failed to retrieve HTML. Status code: {response.status_code}")
