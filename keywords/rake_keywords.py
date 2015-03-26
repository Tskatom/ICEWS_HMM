__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'

from rake import rake
import json

test2 = True

def extract(text, stopfile='/home/weiwang/PycharmProjects/ICEWS_HMM/keywords/rake/SmartStoplist.txt'):
    rake_obj = rake.Rake(stopfile, 4, 2, 1)
    keywords = rake_obj.run(text)
    return [k[0] for k in keywords]


def icews_keywords(file_name):
    #keywords = []
    with open(file_name) as icf:
        for line in icf:
            day_r = json.loads(line)
            for e in day_r['events']:
                text = e['Event Sentence']
                words = extract(text)
                print words
                print text
                print '\n'

if test2:
    file_name = "/home/weiwang/workspace/data/icews_gsr_events/231/14/Montevideo"
    icews_keywords(file_name)