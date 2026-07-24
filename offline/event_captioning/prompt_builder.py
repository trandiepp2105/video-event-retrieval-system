class CaptionPromptBuilder:
    def build(self, subtitle_block: str) -> str:
        return f"""
You are given a short movie event as a video clip.

You also receive structured subtitles that overlap this event.
Each subtitle item has a relative time inside the event and its original text.

Structured subtitles:
{subtitle_block}

Your task:
Generate exactly one Vietnamese retrieval caption for this video event.

The caption will be encoded into a single vector for video event retrieval.
Therefore, it must contain concrete, searchable, human-memorable details.

Main retrieval goal:
- The caption should help match user queries about this event.
- Focus on concrete details that a viewer can remember and later search for.
- Do not produce a beautiful literary caption.
- Do not produce a vague summary.
- Do not produce a high-level interpretation of the scene.
- Prefer visually grounded and subtitle-grounded details over abstract descriptions.

Scene context rule:
- Describe the scene context in more detail than a generic place label.
- Do not write only "trong văn phòng", "trong phòng", "ở nhà hàng", or "trên đường" if more background details are visible.
- Mention the concrete type of place and visible background elements.
- Include background objects that help retrieval, such as computer monitors, desks, counters, shelves, doors, windows, wall colors, signs, plants, chairs, tables, lamps, vehicles, food stalls, kitchen equipment, river, beach sand, street lights, buildings, or shop displays.
- If the place looks like a reception counter, service desk, office lobby, restaurant table, kitchen counter, hospital room, street corner, riverbank, or beach, describe it specifically in Vietnamese.
- The caption should include at least two concrete scene/background details if they are visible.
- Prefer "ở quầy lễ tân trong văn phòng có máy tính và bàn trắng" over only "trong văn phòng".

Subject detail rule:
- Include the main subjects: people, objects, animals, vehicles, or important visual entities.
- If visible, mention useful subject attributes such as clothing color, clothing type, clothing pattern, object color, gender/age group, or distinctive appearance.
- Prefer concrete Vietnamese phrases such as "người đàn ông mặc áo sơ mi caro", "người đàn ông mặc áo ba lỗ đen", "người phụ nữ mặc áo đỏ", "một nhóm người", "chiếc xe màu đen".
- Avoid vague pronouns such as "anh ấy", "cô ấy", or "họ" when the subject can be described more clearly.
- If multiple people appear, distinguish them using visible attributes such as clothing, position, action, or object they hold.
- Do not invent names, identities, family relationships, jobs, or social roles unless they are clearly stated in the subtitles or visible in the scene.

Action and interaction detail rule:
- Describe the actions of the main subjects in more detail, not only the general activity.
- Do not write only generic actions such as "nói chuyện", "đứng", "ngồi", or "đi lại" if more specific actions are visible.
- Identify the action owner clearly: who is doing the action, who is receiving the action, and what object is involved.
- For interactions between people, describe the interaction direction and roles, such as who gives, who receives, who looks at whom, who talks to whom, who follows whom, who helps whom, or who reacts to whom.
- If one person gives or hands an object to another person, explicitly mention the giver, the receiver, and the object.
- If a person is holding, carrying, eating, drinking, opening, pointing, reaching, picking up, putting down, or taking something, include that object if visible.
- If multiple actions happen in sequence, preserve the temporal order using Vietnamese connectors such as "đang", "rồi", "sau đó", "trong lúc", or "khi".
- If two actions happen at the same time, describe the simultaneity using phrases such as "trong lúc", "vừa ... vừa ...", or "khi".
- Include visible body actions that help retrieval, such as turning toward someone, looking at someone, reaching out a hand, passing an object, receiving an object, leaning on a counter, sitting down, standing up, walking away, entering, or leaving the scene.
- Include facial expressions only if they are clearly visible as concrete expressions, such as "mỉm cười", "cau mày", or "nhìn ngạc nhiên"; do not infer abstract emotions.
- If an action is uncertain, use cautious Vietnamese wording such as "có vẻ đang đưa", "dường như đang nhận", or "một vật giống như hộp đồ ăn".
- The caption should mention at least one concrete interaction between subjects if an interaction is visible.
- Prefer detailed action phrases such as "người đàn ông mặc áo ba lỗ đen đưa ly mì cho người đàn ông mặc áo sơ mi caro" over generic phrases such as "hai người nói chuyện".
- Prefer "người đàn ông mặc áo sơ mi caro cầm mũ bảo hiểm đứng trước quầy và nhận ly mì" over "người đàn ông đứng trong văn phòng".

Object detail rule:
- Include important visible objects involved in the action, such as helmet, cup, bowl, food container, phone, bag, document, money, weapon, tool, vehicle, food, bottle, glass, plate, book, computer, or key.
- If an object is central to the interaction, mention it even if the exact object type is uncertain.
- If an object is uncertain, use a cautious generic Vietnamese description such as "một cốc đồ ăn", "một hộp đồ ăn", "một vật nhỏ", or "một món đồ" instead of a wrong specific object.
- Do not over-specify an object if it is not clearly visible.

Subtitle usage rule:
- Use subtitle content when it contains specific searchable information, such as names, jobs, places, plans, requests, objects, tasks, or memorable dialogue.
- Integrate useful subtitle content directly into the Vietnamese caption.
- Do not merely write "họ đang nói chuyện" if the subtitles reveal what they are talking about.
- Do not summarize subtitles abstractly.
- Prefer concrete subtitle facts or phrases, such as "trưởng thôn gọi", "chương trình truyền hình", "làm nhân viên an toàn", "Tôi đi lấy nước nhé", or other specific dialogue from the subtitles.
- If the subtitles explain the conversation topic, combine the visible interaction with the spoken content in the same sentence.
- Do not write vague phrases like "lời thoại thể hiện sự thúc giục", "họ nói về một công việc", "họ bàn chuyện gì đó", or "cuộc trò chuyện có vẻ quan trọng" if the subtitles provide concrete content.
- If the subtitles are not useful for identifying the event, prioritize the visible video content.

Do-not-include rule:
- Do not describe abstract mood, atmosphere, rhythm, cinematic style, or tension.
- Avoid vague phrases such as "không khí căng thẳng", "cuộc trò chuyện cảm xúc", "bầu không khí thân mật", "khoảnh khắc kịch tính", "nhịp điệu chậm rãi", or "tình huống nghiêm trọng".
- Do not invent names, identities, relationships, emotions, intentions, or off-screen facts.
- Do not mention frame numbers, timestamps, camera cuts, shot transitions, or that this is a video clip.
- Do not mention uncertainty unless it is needed to avoid a wrong object/action description.
- Do not output explanations, bullet points, analysis, or any text outside JSON.

Length and format:
- Write exactly one Vietnamese sentence.
- The sentence should be concise but information-rich.
- Recommended length: 45 to 90 Vietnamese words.
- Return valid JSON only.

Output schema:
{{
  "retrieval_caption": "Một câu tiếng Việt duy nhất chứa chủ thể cụ thể, trang phục hoặc ngoại hình, hành động chính, tương tác giữa các chủ thể, tương tác với đồ vật, bối cảnh chi tiết, các chi tiết nền nhìn thấy được và nội dung subtitle nổi bật nếu hữu ích."
}}
""".strip()
