import pandas as pd
from xmlrpclib import ServerProxy
import json
import os
import urllib
import gzip
from itertools import groupby
import re
import unicodedata


def get_subtitles(movies_path='data/ml-1m/processed/movies-enhanced.csv',
                  sub_raw_path='data/subs/raw',
                  opensub_responses_path='data/opensub_responses.csv',
                  search=True, download=True):
    movies = pd.read_csv(movies_path)
    opensub_responses = pd.DataFrame.from_csv(opensub_responses_path)

    # search for subtitles on opensubtitles.org
    if search:
        osd_user = 'marcuniq'
        osd_pw = os.environ.get('OPENSUBTITLE_PW')
        osd_language = 'en'
        osd_server = ServerProxy('http://api.opensubtitles.org/xml-rpc')
        session = osd_server.LogIn(osd_user, osd_pw, osd_language, 'opensubtitles-download 3.5')

        for i in movies.index:
            imdb_id = movies.ix[i]['imdb_id']
            if imdb_id in opensub_responses['imdb_id'].values:
                continue
            search_query = []
            search_query.append({'imdbid': str(imdb_id), 'sublanguageid': 'eng'})
            print 'searching for imdb_id ', imdb_id, ' (', movies.ix[i]['title'], ')'
            search_result = osd_server.SearchSubtitles(session['token'], search_query)

            if search_result['status'] != '200 OK':
                print 'searching FAILED for imdb_id ', imdb_id
                continue

            result_json = json.dumps(search_result)

            # save response
            opensub_responses = opensub_responses.append(pd.DataFrame({'imdb_id': [imdb_id], 'response': [result_json]}))
            opensub_responses.to_csv(opensub_responses_path)

    # download subtitle with highest download count
    if download:
        nb_no_sub = 0
        nb_no_srt = 0
        for i, row in opensub_responses.iterrows():

            imdb_id = str(int(row['imdb_id']))

            sub_fname = imdb_id + '.srt.gz'
            sub_local_path = os.path.join(sub_raw_path, sub_fname)

            if os.path.isfile(sub_local_path):
                print '[info] subtitle already downloaded for ', imdb_id
                continue

            response = json.loads(row['response'])
            data = response['data']
            if len(data) == 0:
                print '[warn] no subtitle found for ', imdb_id
                nb_no_sub += 1
                continue

            # only_ascii = [s for s in data if s['SubEncoding'] == 'ASCII']
            only_srt = [s for s in data if s['SubFormat'] == 'srt']
            if len(only_srt) == 0:
                print '[warn] no srt found for ', imdb_id
                nb_no_srt += 1
                continue

            sub_max_download_count = max(only_srt, key=lambda s: int(s['SubDownloadsCnt']))
            sub_url = sub_max_download_count['SubDownloadLink']

            print '[info] download subtitle for ', imdb_id
            try:
                urllib.urlretrieve(sub_url, sub_local_path)
            except IOError:
                print '[error] failed to download ', imdb_id
                continue

            with gzip.open(sub_local_path) as f:
                try:
                    content = f.read()
                except IOError:
                    f.close()
                    os.remove(sub_local_path)
                    print '[error] couldnt open gzip for ', imdb_id
                    break

        print 'number of subs with empty search results: ', nb_no_sub
        print 'number of subs with no srt format: ', nb_no_srt


def parse_subtitles(subs_path='data/subs/', overwrite=True):
    raw_path = os.path.join(subs_path, 'raw')
    processed_path = os.path.join(subs_path, 'processed')

    files = os.listdir(raw_path)

    for filename in files:
        fname_srt, _ = os.path.splitext(filename)
        imdb_id, _ = os.path.splitext(fname_srt)

        text_path = os.path.join(processed_path, imdb_id + '.txt')
        if os.path.isfile(text_path):
            print "[Info]", text_path, "already processed"
            if overwrite:
                continue

        with gzip.open(os.path.join(raw_path, filename)) as f:
            try:
                res = [list(g) for b,g in groupby(f, lambda x: bool(x.strip())) if b]
            except IOError:
                f.close()
                print "[Error]", filename

        text = []
        print "[Info] processing", filename

        # a section is e.g.
        # [u'1\n',
        # u'00:00:25,959 --> 00:00:30,896\n',
        # u'[Boy]All right, everyone! This...\n',
        # u'is a stickup! Don't anybody move!\n']
        for section in res:
            section_text = ''
            for line in section[2:]:
                section_text += line + ' '

            dismiss = True if len(re.findall(r"subtitle", section_text, flags=re.IGNORECASE)) > 0 else False
            if dismiss:
                continue

            try:
                section_text = section_text.decode('utf-8')
                section_text = unicodedata.normalize('NFKD', section_text).encode('ascii','ignore')
            except UnicodeDecodeError:
                print "[Warn] cant decode as utf-8 file", filename
                pass

            # clean text
            section_text = re.sub(r"\n|\r|\[.*?\]|<.+?>|[*-]|\"", "", section_text)
            section_text = re.sub(r"[^a-zA-Z0-9,!'\.\? ]", "", section_text) # remove non common characters
            # section_text = re.sub(r"[^\x00-\x7f]", "", section_text) # remove non-ascii characters
            section_text = re.sub(r"\.{2,}", ".", section_text) # remove all ...
            section_text = section_text.lower().lstrip('!.').strip()

            text.append(section_text)

        text_to_save = ''
        for t in text:
            text_to_save += t + ' '

        text_to_save = re.sub(r" {2,}", " ", text_to_save) # remove double spaces
        text_to_save = text_to_save.strip()

        with open(text_path, 'wt') as t:
            t.write(text_to_save)


def find_sub_max_download(data, imdb_id):
    only_srt = [s for s in data if s['SubFormat'] == 'srt']
    if len(only_srt) == 0:
        print '[warn] no srt found for ', imdb_id

    sub_max_download_count = max(only_srt, key=lambda s: int(s['SubDownloadsCnt']))
    return sub_max_download_count


def print_non_srt(imdb_id, data):
    sub_max_download_count = find_sub_max_download(data, imdb_id)
    if sub_max_download_count['SubFormat'] != 'srt':
        print imdb_id


def print_sub_rating(imdb_id, data):
    sub_max_download_count = find_sub_max_download(data, imdb_id)
    print imdb_id, sub_max_download_count['SubRating']


def check_local_subs(sub_raw_path='data/subs/raw',
                     opensub_responses_path='data/opensub_responses.csv'):

    opensub_responses = pd.DataFrame.from_csv(opensub_responses_path)
    downloaded_subs = [os.path.join(sub_raw_path, x) for x in os.listdir(sub_raw_path)]

    for sub_path in downloaded_subs:
        _, fname_srt_gz = os.path.split(sub_path)
        fname_srt, _ = os.path.splitext(fname_srt_gz)
        imdb_id, _ = os.path.splitext(fname_srt)

        response = opensub_responses[opensub_responses['imdbId'] == int(imdb_id)]['response'].values[0]
        response = json.loads(response)
        data = response['data']
        print_sub_rating(imdb_id, data)
        #print_non_srt(imdb_id, data)


if __name__ == '__main__':
    parse_subtitles()
