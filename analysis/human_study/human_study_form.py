import glob
import os
import random
import subprocess
import tempfile
from typing import List

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from tqdm import tqdm

import subprocess
from typing import List

from analysis.human_study.assets import form_description, MAIN_ALGORITHMS, algorithms, environments_info, \
    SERVICE_ACCOUNT_FILE, SCOPES


def add_label_to_video(input_video: str, label: str, output_video: str) -> None:
    """Adds a text label at the top-left corner of a video and trims it to 10 seconds."""
    filter_text = f"drawtext=text='{label}':x=10:y=10:fontsize=112:fontcolor=red:box=1:boxcolor=black@0.5"
    cmd = [
        "ffmpeg", "-i", input_video, "-vf", filter_text, "-t", "10", "-codec:a", "copy", "-y", output_video
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)


def concatenate_videos(video_paths: List[str], output_video: str) -> None:
    """Concatenates multiple videos side by side, ensuring each is trimmed to 10 seconds."""
    trimmed_videos = []

    for i, video in enumerate(video_paths):
        trimmed_video = f"trimmed_{i}.mp4"
        cmd = ["ffmpeg", "-i", video, "-t", "10", "-y", trimmed_video]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        trimmed_videos.append(trimmed_video)

    inputs = " ".join(f"-i {v}" for v in trimmed_videos)
    filter_complex = f"hstack=inputs={len(trimmed_videos)}"
    cmd = f"ffmpeg {inputs} -filter_complex {filter_complex} -y {output_video}".split()
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)


def video_to_gif(input_video: str, output_gif: str, fps=30) -> None:
    """Converts a video to a GIF efficiently, ensuring it's 10 seconds max."""

    cmd = [
        "ffmpeg", "-i", input_video, "-t", "10", "-vf", f"fps={fps},scale=640:-1", "-y", output_gif
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)


def create_video_comparisons(MAIN_ALGORITHMS: List[str], N: int, QUESTIONS_PER_ENV: int, VIDEO_FOLDER: str,
                             algorithms: dict, environments_info: dict, OUTPUT_CSV: str) -> pd.DataFrame:
    """Creates video comparisons, shuffling algorithms and videos, and saves to a DataFrame."""
    video_data = []

    for env, info in environments_info.items():
        for _ in range(QUESTIONS_PER_ENV):
            main_algo = random.choice([x for x in MAIN_ALGORITHMS if x in info['algorithms']])
            other_algos = random.sample([a for a in info['algorithms'] if a not in MAIN_ALGORITHMS], N - 1)
            selected_algos = [main_algo] + other_algos
            random.shuffle(selected_algos)

            question_videos = []
            labels = []

            for i, algo in enumerate(selected_algos):
                index = random.randint(1, 10)
                video_path = os.path.join(VIDEO_FOLDER, f"{env}_{algorithms[algo]}_2", f"{index}.mp4")
                if os.path.exists(video_path):
                    question_videos.append(video_path)
                    labels.append(chr(65 + i))  # A, B, C
                else:
                    print(f"Video not found: {video_path}")

            if len(question_videos) == N:
                video_data.append({
                    'env': env,
                    'algorithms': selected_algos,
                    'labels': labels,
                    'video_paths': question_videos
                })

    video_df = pd.DataFrame(video_data)
    video_df.to_csv(OUTPUT_CSV, index=False)
    return video_df


def save_gifs_and_order(GIF_FOLDER: str, video_df: pd.DataFrame) -> None:
    """Saves GIFs and orders them based on the DataFrame."""
    os.makedirs(GIF_FOLDER, exist_ok=True)
    question_idx = 1
    for env_idx, (env, group) in tqdm(enumerate(video_df.groupby('env', sort=False)),
                                      total=len(video_df['env'].unique()),
                                      desc="Creating GIFs for each environment", position=0):
        for i, question_info in tqdm(group.iterrows(),
                                     total=len(group),
                                     desc=f"Creating GIFs for {env}", position=1, leave=False):
            gif_filename = f"Q_{question_idx}__{env}__{'_'.join(question_info['algorithms'])}.gif"
            gif_path = os.path.join(GIF_FOLDER, gif_filename)
            create_gif(question_info['video_paths'], question_info['labels'], gif_path)
            question_idx += 1


def create_gif(video_paths: List[str], labels: List[str], output_gif: str) -> None:
    """Takes mp4 videos, adds labels, concatenates them side by side, and outputs a GIF."""
    assert len(video_paths) == len(labels), "Number of videos must match number of labels"

    with tempfile.TemporaryDirectory() as temp_dir:
        labeled_videos = [os.path.join(temp_dir, f"labeled_{i}.mp4") for i in range(len(video_paths))]

        # Add labels to videos
        for input_video, label, output_video in zip(video_paths, labels, labeled_videos):
            add_label_to_video(input_video, label, output_video)

        # Concatenate videos
        concatenated_video = os.path.join(temp_dir, "concatenated.mp4")
        concatenate_videos(labeled_videos, concatenated_video)

        # Convert to GIF
        video_to_gif(concatenated_video, output_gif)


def create_form(video_df: pd.DataFrame, environments_info: dict, service_account_file: str, scopes: list,
                your_email: str, questions_per_env: int):
    # Authenticate and create the Google Forms API service
    credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=scopes)
    forms_service = build("forms", "v1", credentials=credentials)

    # Create an empty form
    form_metadata = {"info": {"title": "Humanoid Control: Video Comparisons"}}
    form = forms_service.forms().create(body=form_metadata).execute()
    form_id = form["formId"]

    # Update form description
    _update_form_description(form_id, forms_service)

    # Initialize the list to hold form items
    form_items = []

    # Group the DataFrame by environment
    grouped = video_df.groupby('env', sort=False)

    question_num = 1
    for env_idx, (env, group) in enumerate(grouped):
        # Add a page break for each environment
        page_break = {
            "title": environments_info[env]["page_title"],
            "pageBreakItem": {}
        }
        form_items.append(page_break)

        # Iterate over the grouped DataFrame in chunks of 'questions_per_env'
        for question_idx, (_, row) in enumerate(group.iterrows()):
            question_text = f"Q {question_num}: Which example looks more human-like for task **{environments_info[env]['description']}**?"
            question_item = {
                "title": question_text,
                "questionItem": {
                    "question": {
                        "required": True,
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": [{"value": label} for label in row['labels']],
                        },
                    }
                },
            }
            form_items.append(question_item)
            question_num += 1

    # Prepare the batch update request
    update_request = {
        "requests": [{"createItem": {"item": item, "location": {"index": idx}}} for idx, item in enumerate(form_items)]
    }

    # Execute the batch update
    forms_service.forms().batchUpdate(formId=form_id, body=update_request).execute()

    # Share the form with the specified email
    drive_service = build("drive", "v3", credentials=credentials)
    permission = {
        "type": "user",
        "role": "writer",
        "emailAddress": your_email
    }
    drive_service.permissions().create(
        fileId=form_id,
        body=permission,
        sendNotificationEmail=False
    ).execute()

    form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    print("Form updated successfully!")
    print("Form URL:", form_url)
    print(f"Edit permissions granted to {your_email}")


def _update_form_description(form_id, forms_service):
    description_update_request = {
        "requests": [
            {
                "updateFormInfo": {
                    "info": {
                        "description": form_description
                    },
                    "updateMask": "description"
                }
            }
        ]
    }
    forms_service.forms().batchUpdate(formId=form_id, body=description_update_request).execute()


def upload_gifs_to_drive(GIF_FOLDER, DRIVE_FOLDER_ID, SERVICE_ACCOUNT_FILE, SCOPES):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    drive_service = build("drive", "v3", credentials=credentials)

    gif_paths = glob.glob(os.path.join(GIF_FOLDER, "*.gif"))
    for gif_path in tqdm(gif_paths, desc="Uploading GIFs"):
        upload_to_drive(gif_path, DRIVE_FOLDER_ID, drive_service)

    print("GIFs uploaded to Google Drive successfully!")


def upload_to_drive(file_path, drive_folder_id, drive_service):
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


def create_gifs_and_form(drive_folder_id, gif_folder, main_algorithms, n, output_csv, questions_per_env, scopes,
                         service_account_file, video_folder, algorithms, environments_info):
    video_df = create_video_comparisons(main_algorithms, n, questions_per_env, video_folder,
                                        algorithms,
                                        environments_info, output_csv)
    save_gifs_and_order(gif_folder, video_df)
    upload_gifs_to_drive(gif_folder, drive_folder_id, service_account_file, scopes)
    # === Step 3: Create Google Form ===
    create_form(video_df, environments_info, service_account_file, scopes, your_email="your_email",
                questions_per_env=questions_per_env)


def create_single_form():
    # ==== CONFIG ====
    VIDEO_FOLDER = "path/to/input/videos"  # Folder containing input videos
    GIF_FOLDER = "path/to/gif/folder"  # Folder to save GIFs
    OUTPUT_CSV = "shuffled_labels.csv"  # Track label order
    QUESTIONS_PER_ENV = 8  # Number of questions per environment
    N = 3  # Number of total algorithms per question (1 main + N-1 others)
    DRIVE_FOLDER_ID = "ENTER_DRIVE_FOLDERID_HERE"  # Folder to upload GIFs to
    create_gifs_and_form(DRIVE_FOLDER_ID, GIF_FOLDER, MAIN_ALGORITHMS, N, OUTPUT_CSV, QUESTIONS_PER_ENV, SCOPES,
                         SERVICE_ACCOUNT_FILE, VIDEO_FOLDER, algorithms, environments_info)


def create_multi_form():
    # ==== CONFIG ====
    VIDEO_FOLDER = "path/to/video/folder"  # Folder containing input videos
    OUT_DIR = "path/to/out/dir"
    os.makedirs(OUT_DIR, exist_ok=True)
    NUM_FORMS = 3
    GIF_FOLDERS = [os.path.join(OUT_DIR, f"gifs_new_form{i}") for i in range(NUM_FORMS)]  # Folder to save GIFs
    OUTPUT_CSVS = [os.path.join(OUT_DIR, f"shuffled_labels_form{i}.csv") for i in range(NUM_FORMS)]  # Track label order
    QUESTIONS_PER_ENV = 8  # Number of questions per environment
    N = 3  # Number of total algorithms per question (1 main + N-1 others)
    DRIVE_FOLDER_IDS = ["FORM1_DRIVE_FOLDER_ID",  # form 1
                        "FORM2_DRIVE_FOLDER_ID",  # form 2
                        "FORM3_DRIVE_FOLDER_ID"]  # form 3

    for GIF_FOLDER, OUTPUT_CSV, DRIVE_FOLDER_ID in zip(GIF_FOLDERS, OUTPUT_CSVS, DRIVE_FOLDER_IDS):
        create_gifs_and_form(DRIVE_FOLDER_ID, GIF_FOLDER, MAIN_ALGORITHMS, N, OUTPUT_CSV, QUESTIONS_PER_ENV, SCOPES,
                             SERVICE_ACCOUNT_FILE, VIDEO_FOLDER, algorithms, environments_info)


def main():
    create_multi_form()
    # create_single_form()


if __name__ == '__main__':
    main()
