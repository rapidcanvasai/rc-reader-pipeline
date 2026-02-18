# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Reading data from multiple folders and copying to a new folder
# Required imports
from utils.notebookhelpers.helpers import Helpers
import shutil
import os

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

print(f"Context : {context}")
token = Helpers.get_user_token(context)
print(f"Token: {token}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Download artifacts from both folders
try:
    # Download files from "Historical Data" folder
    historical_data_files = Helpers.downloadArtifacts(context, 'Historical Data')
    print(f"Files from 'Historical Data': {list(historical_data_files.keys())}")
except Exception as e:
    print(f"Error downloading 'Historical Data': {e}")
    historical_data_files = {}

try:
    # Download files from "Historical-Data" folder
    historical_data_files_2 = Helpers.downloadArtifacts(context, 'Historical-Data')
    print(f"Files from 'Historical-Data': {list(historical_data_files_2.keys())}")
except Exception as e:
    print(f"Error downloading 'Historical-Data': {e}")
    historical_data_files_2 = {}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create new artifacts directory
artifactsDir = Helpers.getOrCreateArtifactsDir(context, artifactsId="Consolidated-Historical-Data")

# Copy files from "Historical Data"
for filename, filepath in historical_data_files.items():
    try:
        destination = os.path.join(artifactsDir, filename)
        shutil.copy(filepath, destination)
        print(f"Copied {filename} from 'Historical Data'")
    except Exception as e:
        print(f"Error copying {filename}: {e}")

# Copy files from "Historical-Data"
for filename, filepath in historical_data_files_2.items():
    try:
        destination = os.path.join(artifactsDir, filename)
        shutil.copy(filepath, destination)
        print(f"Copied {filename} from 'Historical-Data'")
    except Exception as e:
        print(f"Error copying {filename}: {e}")

print(f"\nAll files copied to 'Consolidated-Historical-Data' folder")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Add artifact to output
Helpers.save_output_artifacts(context=context, artifact_name='Consolidated-Historical-Data')