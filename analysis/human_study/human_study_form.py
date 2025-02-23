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


def create_video_comparisons(MAIN_ALGORITHMS, N, QUESTION_NUM, VIDEO_FOLDER, algorithms, environments_info, OUTPUT_CSV):
    video_pairs = []
    shuffling_info = []
    for env, info in environments_info.items():
        for _ in range(QUESTION_NUM):
            main_algo = random.choice([x for x in MAIN_ALGORITHMS if x in info['algorithms']])
            other_algos = random.sample([a for a in info['algorithms'] if a not in MAIN_ALGORITHMS], N - 1)
            selected_algos = [main_algo] + other_algos
            random.shuffle(selected_algos)

            video_paths = []
            labels = []
            for i, algo in enumerate(selected_algos):
                index = random.randint(1, 10)
                video_path = os.path.join(VIDEO_FOLDER, f"{env}_{algorithms[algo]}_0", f"{index}.mp4")
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(chr(65 + i))  # A, B, C
                else:
                    print(f"Video not found: {video_path}")

            shuffling_info.append([env] + selected_algos + labels)
            video_pairs.append((env, selected_algos, video_paths, labels))
    columns = ["Environment"] + [f"Algo{i}" for i in range(N)] + [f"Label{i}" for i in range(N)]
    pd.DataFrame(shuffling_info,
                 columns=columns).to_csv(
        OUTPUT_CSV, index=True)
    return shuffling_info, video_pairs


def save_gifs_and_order(GIF_FOLDER, video_pairs):
    # Save label mappings
    os.makedirs(GIF_FOLDER, exist_ok=True)
    # === Step 2: Convert Videos to GIFs ===
    for i, (env, selected_algos, video_paths, labels) in enumerate(tqdm(video_pairs, desc="Creating GIFs")):
        gif_filename = f"Q_{i + 1}__{env}__{'_'.join(selected_algos)}.gif"
        gif_path = os.path.join(GIF_FOLDER, gif_filename)
        create_gif(video_paths, labels, gif_path)


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


def create_form(video_pairs, environments_info, SERVICE_ACCOUNT_FILE, SCOPES, YOUR_EMAIL):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    # Create Google Forms API service
    forms_service = build("forms", "v1", credentials=credentials)

    # Create an empty form
    form_metadata = {"info": {"title": "Human Study: Video Comparisons"}}
    form = forms_service.forms().create(body=form_metadata).execute()
    form_id = form["formId"]

    # Add questions in pages of 15
    questions = []
    for i, (env, selected_algos, video_paths, labels) in enumerate(video_pairs):
        if i % 15 == 0:
            # Create a new page for every 15 questions
            page_break = {
                "title": f"Page {i // 15 + 1}",
                "pageBreakItem": {}
            }
            questions.append(page_break)

        question_text = \
            f"Q {i + 1}: Which example looks more human-like for task **{environments_info[env]['description']}**?"
        questions.append(
            {
                "title": question_text,
                "questionItem": {
                    "question": {
                        "required": True,
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": [{"value": label} for label in labels],
                        },
                    }
                },
            }
        )

    update_request = {
        "requests": [{"createItem": {"item": q, "location": {"index": i}}} for i, q in enumerate(questions)]
    }
    forms_service.forms().batchUpdate(formId=form_id, body=update_request).execute()

    # Create Google Drive API service
    drive_service = build("drive", "v3", credentials=credentials)

    # Share the form with your email
    permission = {
        "type": "user",
        "role": "writer",
        "emailAddress": YOUR_EMAIL
    }
    drive_service.permissions().create(
        fileId=form_id,
        body=permission,
        sendNotificationEmail=False
    ).execute()

    form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    print("Form updated successfully!")
    print("Form URL:", form_url)
    print(f"Edit permissions granted to {YOUR_EMAIL}")


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


def main():
    # ==== CONFIG ====
    VIDEO_FOLDER = "../../output/FINALLY_"  # Folder containing input videos
    # VIDEO_FOLDER = "../../output/old_FINALLY_"  # Folder containing input videos
    GIF_FOLDER = "gifs_new"  # Folder to save GIFs
    OUTPUT_CSV = "shuffled_labels.csv"  # Track label order
    QUESTIONS_PER_ENV = 15  # Number of questions per environment
    N = 3  # Number of total algorithms per question (1 main + N-1 others)

    SCOPES = ["https://www.googleapis.com/auth/forms.body", "https://www.googleapis.com/auth/drive.file"]
    SERVICE_ACCOUNT_FILE = "service_account_secret.json"

    DRIVE_FOLDER_ID = "1r241VD5rwHhrigHtF56LQC2c3FgwhkER"  # Folder to upload GIFs to

    MAIN_ALGORITHMS = [
        "MaskedMimic_Inversion_Prior_False",
        "MaskedMimic_Inversion_Prior_True",
    ]

    algorithms = {
        "MaskedMimic_FineTune_Prior_True": "prior_True_text_False_current_pose_True_bigger_True_train_actor_True",
        "MaskedMimic_FineTune_Prior_False": "prior_False_text_False_current_pose_True_bigger_True_train_actor_True",
        "MaskedMimic_Inversion_Prior_True": "prior_True_text_False_current_pose_True_bigger_True_train_actor_False",
        "MaskedMimic_Inversion_Prior_False": "prior_False_text_False_current_pose_True_bigger_True_train_actor_False",
        "AMP": "disable_discriminator_False",
        "PPO": "disable_discriminator_True",
        "PULSE": "pulse",
        "MaskedMimic_Prior_Only": "prior_True_text_False_current_pose_True_bigger_True_train_actor_False_prior_only"
    }

    environments_info = {
        "steering": {
            "description": "walking in red direction",
            "algorithms": [
                "MaskedMimic_FineTune_Prior_True",
                "MaskedMimic_FineTune_Prior_False",
                "MaskedMimic_Inversion_Prior_True",
                "MaskedMimic_Inversion_Prior_False",
                "AMP",
                "PPO",
                "PULSE",
                "MaskedMimic_Prior_Only"
            ]
        },
        "direction_facing": {
            "description": "walking in red direction, looking at the blue direction",
            "algorithms": [
                "MaskedMimic_FineTune_Prior_True",
                "MaskedMimic_FineTune_Prior_False",
                "MaskedMimic_Inversion_Prior_True",
                "MaskedMimic_Inversion_Prior_False",
                "AMP",
                "PPO",
                "PULSE",
                "MaskedMimic_Prior_Only"
            ]
        },
        "reach": {
            "description": "reaching for the dot",
            "algorithms": [
                "MaskedMimic_FineTune_Prior_True",
                "MaskedMimic_FineTune_Prior_False",
                "MaskedMimic_Inversion_Prior_True",
                "MaskedMimic_Inversion_Prior_False",
                "AMP",
                "PPO",
                "PULSE",
                "MaskedMimic_Prior_Only"
            ]
        },
        "strike": {
            "description": "walking and hitting the target",
            "algorithms": [
                "MaskedMimic_FineTune_Prior_False",
                "MaskedMimic_Inversion_Prior_False",
                "AMP",
                "PPO",
                "PULSE"
            ]
        },
        "long_jump": {
            "description": "running and jumping",
            "algorithms": [
                "MaskedMimic_FineTune_Prior_False",
                "MaskedMimic_Inversion_Prior_False",
                "AMP",
                "PPO",
                "PULSE"
            ]
        }
    }

    shuffling_info, video_pairs = create_video_comparisons(MAIN_ALGORITHMS, N, QUESTIONS_PER_ENV, VIDEO_FOLDER,
                                                           algorithms,
                                                           environments_info, OUTPUT_CSV)

    save_gifs_and_order(GIF_FOLDER, video_pairs)

    upload_gifs_to_drive(GIF_FOLDER, DRIVE_FOLDER_ID, SERVICE_ACCOUNT_FILE, SCOPES)

    # === Step 3: Create Google Form ===
    create_form(video_pairs, environments_info, SERVICE_ACCOUNT_FILE, SCOPES, YOUR_EMAIL="rvainshtein@gmail.com")


if __name__ == '__main__':
    main()
