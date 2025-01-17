import requests

# Set the API endpoint URL
api_endpoint = 'http://dm.dinus.ac.id:6003/detection'
# api_endpoint = 'http://localhost:5000/detection'

# Load the image file
image_file = 'image.jpg'

try:
    # Create a dictionary with the image file
    file = {'image': open(image_file, 'rb')}

    # Send the POST request to the API endpoint
    response = requests.post(api_endpoint, files=file)

    # Check if the response was successful (status code 200)
    if response.status_code == 200:
        # Retrieve the JSON response
        json_response = response.json()

        # Process the response as needed
        names = [item['name'] for item in json_response if 'name' in item]
        accuracy = [item['confidence'] for item in json_response if 'confidence' in item]

        # Print the response
        print(json_response)
    else:
        print('Request failed with status code:', response.status_code)

except Exception as e:
    print('An error occurred:', str(e))