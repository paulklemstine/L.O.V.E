import sys

def scout_talent(keywords_str):
    print(f'Searching for talent based on: {keywords_str}')
    keywords = [k.strip().lower() for k in keywords_str.split(',')]
    required_keywords = ['fashion model', 'young adult woman', 'open minded']

    if all(kw in keywords for kw in required_keywords):
        print('Found potential candidate: Anya Petrova (simulated).')
        print('This is a simulated result for demonstration purposes.')
        print('Talent scouting process completed successfully.')
    else:
        print('No suitable talent found for the given keywords.')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        scout_talent(sys.argv[1])
    else:
        print('Usage: python talent_scout.py "keyword1,keyword2,..."')
