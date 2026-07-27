import json
with open('../llama2.c/data/TinyStories_all_data/data00.json') as f:
    data = json.load(f)
text = ' '.join(item['story'] for item in data[:15])
with open('test/grad/g4_text.txt', 'w') as f:
    f.write(text)
print(f'{len(data[:15])} stories, {len(text)} chars')
