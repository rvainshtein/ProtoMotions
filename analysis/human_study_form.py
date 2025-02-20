import os
import random
import pandas as pd
from moviepy.editor import VideoFileClip, clips_array
from googleapiclient.discovery import build
from google.oauth2 import service_account

# ==== CONFIG ====
VIDEO_FOLDER = "../output/recordings/final_recordings"  # Folder containing input videos
GIF_FOLDER = "gifs"  # Folder to save GIFs
DRIVE_FOLDER_ID = "1zl0hEjSnp8lHfWM9QJ0O1lKXItrnfgE0"  # Google Drive folder where GIFs are uploaded

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
SERVICE_ACCOUNT_FILE = "service_account_secret.json"

# Authenticate Google Drive API
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)

# === Step 1: Load Videos & Prepare Pairs ===
environments = [
    "steering",
    "direction_facing",
    "reach",
    # "strike",
    # "long_jump"
]
algorithms = {
    # "MaskedMimic_Prio_Only": "",
    "MaskedMimic_FineTune_Prior_True": "prior_True_text_False_current_pose_True_bigger_True_train_actor_True",
    "MaskedMimic_FineTune_Prior_False": "prior_False_text_False_current_pose_True_bigger_True_train_actor_True",
    "MaskedMimic_Inversion_Prior_True": "prior_False_text_False_current_pose_True_bigger_True_train_actor_False",
    "MaskedMimic_Inversion_Prior_False": "prior_True_text_False_current_pose_True_bigger_True_train_actor_False",
    "AMP": "disable_discriminator_False",
    "PPO": "disable_discriminator_True",
    # "PULSE": "",
}

# Store algorithm pairs and their random placement
video_pairs = []

for env in environments:
    for _ in range(5):  # 5 comparisons per environment
        algo1, algo2 = random.sample(algorithms.keys(), 2)
        index1, index2 = random.randint(1, 5), random.randint(1, 5)

        vid1_path = os.path.join(VIDEO_FOLDER, f"{env}_{algorithms[algo1]}_0/{index1}.mp4")
        vid2_path = os.path.join(VIDEO_FOLDER, f"{env}_{algorithms[algo2]}_0/{index2}.mp4")

        if os.path.exists(vid1_path) and os.path.exists(vid2_path):
            # Randomize left/right order but keep track
            left_algo, right_algo = (algo1, algo2) if random.random() < 0.5 else (algo2, algo1)
            left_vid, right_vid = (vid1_path, vid2_path) if left_algo == algo1 else (vid2_path, vid1_path)

            algo_folder = f"{env}_{algorithms[algo1]}_0"
            gif_filename = f"{algo_folder}__{index1}.gif"
            gif_path = os.path.join(GIF_FOLDER, gif_filename)

            video_pairs.append((env, left_algo, right_algo, left_vid, right_vid, gif_path))

# === Step 2: Convert Video Pairs to GIFs ===
os.makedirs(GIF_FOLDER, exist_ok=True)


def create_gif(video1, video2, output_gif):
    clip1 = VideoFileClip(video1).resize(0.5)  # Resize if needed
    clip2 = VideoFileClip(video2).resize(0.5)

    final_clip = clips_array([[clip1, clip2]])

    # Reduce FPS, optimize color palette, and compress
    final_clip.write_gif(
        output_gif,
        fps=12,  # Lower FPS reduces size
        program="ffmpeg",  # More efficient than ImageMagick
        opt="nq",  # Optimize GIF size
        colors=128  # Reduce color depth
    )


for env, left_algo, right_algo, left_vid, right_vid, gif_path in video_pairs:
    create_gif(left_vid, right_vid, gif_path)


# === Step 3: Upload GIFs to Google Drive ===
from googleapiclient.http import MediaFileUpload


def upload_to_drive(file_path, drive_folder_id):
    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [drive_folder_id]
    }

    media = MediaFileUpload(file_path, mimetype="image/gif")  # Correct MediaFileUpload usage

    file = drive_service.files().create(body=file_metadata, media_body=media).execute()

    # Make file public
    drive_service.permissions().create(
        fileId=file["id"],
        body={"role": "reader", "type": "anyone"}
    ).execute()

    return f"https://drive.google.com/uc?id={file['id']}"


gif_links = []

for env, left_algo, right_algo, _, _, gif_path in video_pairs:
    gif_url = upload_to_drive(gif_path, DRIVE_FOLDER_ID)
    gif_links.append((env, left_algo, right_algo, gif_url))

# Save mapping to CSV (for debugging / reference)
df = pd.DataFrame(gif_links, columns=["Environment", "Left Algo", "Right Algo", "GIF URL"])
df.to_csv("gif_mapping.csv", index=False)

# === Step 4: Insert GIF Links into Google Form ===
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/forms.body"]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
forms_service = build("forms", "v1", credentials=credentials)

# Create Google Form
form_metadata = {"info": {"title": "Human Study: Video Comparisons"}}
form = forms_service.forms().create(body=form_metadata).execute()
form_id = form["formId"]

print(f"Form created: https://forms.google.com/d/{form_id}/edit")

questions = []

for env, left_algo, right_algo, gif_url in gif_links:
    question_text = f"Which video is better for {env}?\n\n"
    question_text += f"![Comparison GIF]({gif_url})\n"

    questions.append(
        {
            "title": question_text,
            "questionItem": {
                "question": {
                    "required": True,
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [
                            {"value": f"Left ({left_algo})"},
                            {"value": f"Right ({right_algo})"},
                            {"value": "Both Equally Good"},
                        ],
                    },
                }
            },
        }
    )

# Add questions to the form
update_request = {"requests": [{"createItem": {"item": q}} for q in questions]}
forms_service.forms().batchUpdate(formId=form_id, body=update_request).execute()

print("Form updated successfully!")
