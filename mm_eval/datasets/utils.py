def short_answer(answer):
    answer = answer.split('\n')[0]
    answer = answer.split('. ')[0]
    answer = answer.split('\"')[0]
    answer = answer.split(', ')[0]
    answer = answer.strip()
    answer = answer.lower()
    answer = answer if len(answer) == 0 or answer[-1] != '.' else answer[:-1]
    answer = answer.replace('it is ', '', 1) if answer.startswith('it is ') else answer
    answer = answer.replace('it\'s ', '', 1) if answer.startswith('it\'s ') else answer
    answer = answer.replace('a ', '', 1) if answer.startswith('a ') else answer
    answer = answer.replace('an ', '', 1) if answer.startswith('an ') else answer
    answer = answer.replace('the ', '', 1) if answer.startswith('the ') else answer
    return answer