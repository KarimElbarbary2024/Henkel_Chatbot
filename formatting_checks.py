import re
import pdfplumber

PDF_PATH = "data/iphone_user_guide.pdf"

GROUND_TRUTH = [
    ("iPhone overview", 8),
    ("Accessories", 9),
    ("Multi-Touch screen", 10),
    ("Buttons", 10),
    ("Status icons", 12),
    ("Install the SIM card", 14),
    ("Set up and activate iPhone", 14),
    ("Connect iPhone to your computer", 15),
    ("Connect to Wi-Fi", 15),
    ("Connect to the Internet", 16),
    ("Set up mail and other accounts", 16),
    ("Apple ID", 16),
    ("Manage content on your iOS devices", 17),
    ("iCloud", 17),
    ("Sync with iTunes", 18),
    ("Date and time", 19),
    ("International settings", 19),
    ("Your iPhone name", 19),
    ("View this user guide on iPhone", 20),
    ("Use apps", 21),
    ("Customize iPhone", 23),
    ("Type text", 25),
    ("Dictate", 28),
    ("Voice Control", 29),
    ("Search", 30),
    ("Control Center", 30),
    ("Alerts and Notification Center", 31),
    ("Sounds and silence", 32),
    ("Do Not Disturb", 32),
    ("AirDrop, iCloud, and other ways to share", 32),
    ("Transfer files", 33),
    ("Personal Hotspot", 33),
    ("AirPlay", 34),
    ("AirPrint", 34),
    ("Use an Apple headset", 35),
    ("Bluetooth devices", 35),
    ("Restrictions", 36),
    ("Privacy", 36),
    ("Security", 37),
    ("Charge and monitor the battery", 39),
    ("Travel with iPhone", 40),
    ("Make requests", 41),
    ("Tell Siri about yourself", 42),
    ("Make corrections", 42),
    ("Siri settings", 43),
    ("Phone calls", 44),
    ("Visual voicemail", 47),
    ("Contacts", 48),
    ("Call forwarding, call waiting, and caller ID", 48),
    ("Ringtones and vibrations", 48),
    ("International calls", 48),
    ("Phone settings", 49),
    ("Write messages", 50),
    ("Get a sneak peek", 51),
    ("Finish a message later", 51),
    ("See important messages", 51),
    ("Attachments", 52),
    ("Work with multiple messages", 53),
    ("See and save addresses", 53),
    ("Print messages", 54),
    ("Mail settings", 54),
    ("Safari at a glance", 55),
    ("Search the web", 55),
    ("Browse the web", 56),
    ("Keep bookmarks", 56),
    ("Share what you discover", 57),
    ("Fill in forms", 57),
    ("Avoid clutter with Reader", 58),
    ("Save a reading list for later", 58),
    ("Privacy and security", 59),
    ("Safari settings", 59),
    ("iTunes Radio", 60),
    ("Get music", 61),
    ("Browse and play", 61),
    ("Album Wall", 63),
    ("Audiobooks", 63),
    ("Playlists", 63),
    ("Genius—made for you", 64),
    ("Siri and Voice Control", 64),
    ("iTunes Match", 65),
    ("Home Sharing", 65),
    ("Music settings", 66),
    ("SMS, MMS, and iMessages", 67),
    ("Send and receive messages", 67),
    ("Manage conversations", 68),
    ("Share photos, videos, and more", 69),
    ("Messages settings", 69),
    ("Calendar at a glance", 70),
    ("Invitations", 71),
    ("Use multiple calendars", 71),
    ("Share iCloud calendars", 72),
    ("Calendar settings", 72),
    ("View photos and videos", 73),
    ("Organize your photos and videos", 74),
    ("iCloud Photo Sharing", 74),
    ("My Photo Stream", 75),
    ("Share photos and videos", 76),
    ("Edit photos and trim videos", 77),
    ("Print photos", 77),
    ("Photos settings", 77),
    ("Camera at a glance", 78),
    ("Take photos and videos", 79),
    ("HDR", 80),
    ("View, share, and print", 80),
    ("Camera settings", 81),
    ("Find places", 86),
    ("Get more info", 87),
    ("Get directions", 87),
    ("3D and Flyover", 88),
    ("Maps settings", 88),
    ("Videos at a glance", 89),
    ("Add videos to your library", 90),
    ("Control playback", 90),
    ("Videos settings", 91),
    ("Notes at a glance", 92),
    ("Use notes in multiple accounts", 93),
    ("Scheduled reminders", 95),
    ("Location reminders", 95),
    ("Reminders settings", 95),
    ("Game Center at a glance", 98),
    ("Play games with friends", 99),
    ("Game Center settings", 99),
    ("Newsstand at a glance", 100),
    ("iTunes Store at a glance", 101),
    ("Browse or search", 101),
    ("Purchase, rent, or redeem", 102),
    ("iTunes Store settings", 102),
    ("App Store at a glance", 103),
    ("Find apps", 104),
    ("Purchase, redeem, and download", 104),
    ("App Store settings", 105),
    ("Passbook at a glance", 106),
    ("Passbook on the go", 106),
    ("Passbook settings", 107),
    ("Compass at a glance", 108),
    ("On the level", 109),
    ("Voice Memos at a glance", 110),
    ("Record", 110),
    ("Listen", 111),
    ("Move recordings to your computer", 111),
    ("FaceTime at a glance", 112),
    ("Make and answer calls", 113),
    ("Manage calls", 113),
    ("Contacts at a glance", 114),
    ("Use Contacts with Phone", 115),
    ("Add contacts", 115),
    ("Contacts settings", 116),
    ("Get iBooks", 118),
    ("Read a book", 118),
    ("Organize books", 119),
    ("Read PDFs", 120),
    ("iBooks settings", 120),
    ("At a glance", 121),
    ("Link and calibrate your sensor", 121),
    ("Work out", 122),
    ("Nike + iPod Settings", 122),
    ("Podcasts at a glance", 123),
    ("Get podcasts", 124),
    ("Control playback", 124),
    ("Organize your podcasts", 125),
    ("Podcasts settings", 125),
    ("Accessibility features", 126),
    ("Accessibility Shortcut", 127),
    ("VoiceOver", 127),
    ("Siri", 137),
    ("Zoom", 138),
    ("Invert Colors", 138),
    ("Speak Selection", 138),
    ("Speak Auto-text", 138),
    ("Large, bold, and high-contrast text", 139),
    ("Reduce onscreen motion", 139),
    ("On/off switch labels", 139),
    ("Hearing aids", 139),
    ("Subtitles and closed captions", 140),
    ("LED Flash for Alerts", 141),
    ("Mono audio and balance", 141),
    ("Call audio routing", 141),
    ("Assignable ringtones and vibrations", 141),
    ("Phone noise cancellation", 141),
    ("Guided Access", 141),
    ("Switch Control", 142),
    ("AssistiveTouch", 145),
    ("TTY support", 146),
    ("Visual voicemail", 146),
    ("Widescreen keyboards", 146),
    ("Large phone keypad", 146),
    ("Voice Control", 146),
    ("Accessibility in OS X", 146),
    ("Use international keyboards", 147),
    ("Special input methods", 148),
    ("Mail, Contacts, and Calendar", 149),
    ("Network access", 149),
    ("Apps", 149),
    ("Important safety information", 151),
    ("Important handling information", 153),
    ("iPhone Support site", 154),
    ("Restart or reset iPhone", 154),
    ("Reset iPhone settings", 154),
    ("Get information about your iPhone", 155),
    ("Usage information", 155),
    ("Disabled iPhone", 155),
    ("Back up iPhone", 156),
    ("Update and restore iPhone software", 157),
    ("Cellular settings", 157),
    ("Sell or give away iPhone?", 158),
    ("Learn more, service, and support", 158),
    ("FCC compliance statement", 159),
    ("Canadian regulatory statement", 159),
    ("Disposal and recycling information", 160),
    ("Apple and the environment", 161),
]


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def build_page_map(pdf):
    page_map = {}
    for idx, page in enumerate(pdf.pages):
        if not page.chars:
            continue
        max_y = max(round(c["top"], 1) for c in page.chars)
        footer_chars = [c for c in page.chars if round(c["top"], 1) == max_y]
        footer_chars.sort(key=lambda c: c["x0"])
        footer_text = "".join(c["text"] for c in footer_chars).strip()
        tokens = footer_text.split()
        if tokens and re.match(r'^\d+$', tokens[-1]):
            printed = int(tokens[-1])
            if printed not in page_map:
                page_map[printed] = idx
    return page_map


def extract_sections(pdf, page_map):
    gt_lookup = {}
    for name, page in GROUND_TRUTH:
        gt_lookup[normalize(name)] = (name, page)

    found = []
    seen = set()

    for idx, page in enumerate(pdf.pages):
        if idx < 7:
            continue
        text = page.extract_text()
        if not text:
            continue

        printed_page = next(
            (p for p, i in page_map.items() if i == idx),
            idx + 1
        )

        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            norm = normalize(stripped)
            if norm not in gt_lookup:
                continue
            gt_name, gt_page = gt_lookup[norm]
            if len(stripped) > len(gt_name) + 15:
                continue
            key = (norm, printed_page)
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "name": gt_name,
                "expected_page": gt_page,
                "found_on_pdf_page": printed_page,
            })

    return found


def run_comparison(found):
    gt_names = {normalize(name) for name, _ in GROUND_TRUTH}
    found_names = {normalize(f["name"]) for f in found}

    missed = [(name, page) for name, page in GROUND_TRUTH if normalize(name) not in found_names]
    wrong_page = [
        f for f in found
        if normalize(f["name"]) in gt_names
        and f["found_on_pdf_page"] != f["expected_page"]
    ]
    extra = [f for f in found if normalize(f["name"]) not in gt_names]

    print("found:", len(found_names), "unique headings /", len(GROUND_TRUTH), "total")
    print("wrong page:", len(wrong_page))
    print("missed:", len(missed))
    print("extra noise:", len(extra))

    if missed:
        print("\nmissed:")
        for name, page in missed:
            print("  expected p" + str(page) + " |", name)

    if wrong_page:
        print("\nwrong page:")
        for f in wrong_page:
            print("  found p" + str(f["found_on_pdf_page"]) + " expected p" + str(f["expected_page"]) + " |", f["name"])

    if extra:
        print("\nextra noise:")
        for f in extra:
            print("  p" + str(f["found_on_pdf_page"]) + " |", f["name"])

    print("\ntrue matches:")
    for f in found:
        if f["found_on_pdf_page"] == f["expected_page"]:
            print("  p" + str(f["found_on_pdf_page"]) + " |", f["name"])


if __name__ == "__main__":
    with pdfplumber.open(PDF_PATH) as pdf:
        print("loaded", len(pdf.pages), "pages")
        page_map = build_page_map(pdf)
        print("page map built,", len(page_map), "pages mapped")
        print()
        found = extract_sections(pdf, page_map)
        print("candidates found:", len(found))
        print()
        run_comparison(found)