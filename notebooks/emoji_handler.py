import re
import emoji

def emoji_extraction(sent):
    e_sent = emoji.demojize(sent)
    return re.findall(':(.*):',e_sent)
def emoji_count(sent):
    e_sent = emoji.demojize(sent)
    return len(re.findall(':(.*):',e_sent))

# Work for single emoji as we will see further there is atmost one emoji is present in a single sentence.
def emoji_to_text(sent):
    e_sent = emoji.demojize(sent)
    emo = re.findall(':(.*):',e_sent)
    if len(emo)==1:
        return re.sub(':(.*):',emo[0],e_sent)
    else:
        return re.sub(':(.*):','',e_sent)