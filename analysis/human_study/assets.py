form_description = """
In this study, you will watch three short videos side by side each time. 

These videos show different ways a character moves to complete a task. Your job is to decide which movement looks the most human-like.

Each video is labeled A, B, or C — the labels are shuffled every time. Just pick the one you think does the best job.

 Don’t worry — there’s no right or wrong answer! We just want your opinion.

במחקר זה, תצפו בשלושה סרטונים קצרים זה לצד זה בכל פעם. 

 התפקיד שלכם הוא להחליט איזו תנועה נראית הכי אנושית.

כל סרטון מסומן A, B או C - אקראית. 

פשוט בחרו את זה שאתם חושבים שנראה הכי טוב. 

אל תדאגו - אין תשובה נכונה או לא נכונה!

תודה רבה 😘
"""

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
        "page_title": "Walking in red direction - הליכה בכיוון האדום",
        "algorithms": [
            "MaskedMimic_FineTune_Prior_True",
            "MaskedMimic_FineTune_Prior_False",
            "MaskedMimic_Inversion_Prior_True",
            # "MaskedMimic_Inversion_Prior_False",
            "AMP",
            "PPO",
            "PULSE",
            "MaskedMimic_Prior_Only"
        ]
    },
    "direction_facing": {
        "description": "walking in red direction, looking at the blue direction",
        "page_title": "Walking in red direction, while facing the blue direction - הליכה בכיוון האדום, תוך פנייה לכיוון הכחול",
        "algorithms": [
            "MaskedMimic_FineTune_Prior_True",
            "MaskedMimic_FineTune_Prior_False",
            "MaskedMimic_Inversion_Prior_True",
            # "MaskedMimic_Inversion_Prior_False",
            "AMP",
            "PPO",
            "PULSE",
            "MaskedMimic_Prior_Only"
        ]
    },
    "reach": {
        "description": "reaching for the dot",
        "page_title": "Reaching the red dot with the right hand - נגיעה עם יד ימין בנקודה האדומה",
        "algorithms": [
            "MaskedMimic_FineTune_Prior_True",
            "MaskedMimic_FineTune_Prior_False",
            "MaskedMimic_Inversion_Prior_True",
            # "MaskedMimic_Inversion_Prior_False",
            "AMP",
            "PPO",
            "PULSE",
            "MaskedMimic_Prior_Only"
        ]
    },
    "strike": {
        "description": "walking and hitting the target",
        "page_title": "Striking the target - הליכה למטרה והפלה שלה",
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
        "page_title": "Running and jumping forward - ריצה וקפיצה למרחק",
        "algorithms": [
            "MaskedMimic_FineTune_Prior_False",
            "MaskedMimic_Inversion_Prior_False",
            "AMP",
            "PPO",
            "PULSE"
        ]
    }
}

SCOPES = ["https://www.googleapis.com/auth/forms.body", "https://www.googleapis.com/auth/drive.file"]
SERVICE_ACCOUNT_FILE = "service_account_secret.json"
