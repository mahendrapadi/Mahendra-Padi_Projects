

from collections import Counter
import json
import requests
import zlib
import csv


BASEURL = 'https://aws-publicdatasets.s3.amazonaws.com/'
INDEX1 = 'common-crawl/cc-index/collections/CC-MAIN-2015-11/indexes/'
#INDEX2 = 'common-crawl/cc-index/collections/CC-MAIN-2016-44/indexes/'
SPLITS = 1
#list=[]
#k=0
def process_index(index):
    total_length = 0
    total_processed = 0
    total_urls = 0
    mime_types = Counter()
    with open("CC_Urls_2015_11.csv",'wb') as f:
        csv_writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)

        for i in range(SPLITS):
            unconsumed_text = ''
            filename = 'cdx-%05d.gz' % i
            url = BASEURL + index + filename
            response = requests.get(url, stream=True)
            length = int(response.headers['content-length'].strip())
            decompressor = zlib.decompressobj(16+zlib.MAX_WBITS)
            total = 0

            for chunk in response.iter_content(chunk_size=2048):
                total += len(chunk)
                if len(decompressor.unused_data) > 0:
                    # restart decompressor if end of a chunk
                    to_decompress = decompressor.unused_data + chunk
                    decompressor = zlib.decompressobj(16+zlib.MAX_WBITS)
                else:
                    to_decompress = decompressor.unconsumed_tail + chunk
                s = unconsumed_text + decompressor.decompress(to_decompress)
                unconsumed_text = ''

                if len(s) == 0:
                    # Not sure why this happens, but doesn't seem to affect things
                    print 'Decompressed nothing %2.2f%%' % (total*100.0/length),\
                        length, total, len(chunk), filename

                for l in s.split('\n'):     # and (k in range(0,10)):
                    list = []
                    pieces = l.split(' ')
                    if len(pieces) < 3 or l[-1] != '}':
                        unconsumed_text = l
                    else:
                        json_string = ' '.join(pieces[2:])
                        try:
                            metadata = json.loads(json_string)
                        except:
                            print 'JSON load failed: ', total, l
                            assert False

                        url = metadata['url']
                        #print url

                        list.append(url.encode('utf-8'))

                        csv_writer.writerow(list)    #.encode('utf-8'))

                        if 'mime' in metadata:
                            mime_types[metadata['mime']] += 1
                        else:
                            mime_types['<none>'] += 1
                            # print 'No mime type for ', url
                        total_urls += 1
                        # print url
                        list.append(url)  # .encode('utf-8')
            print 'Done with ', filename
            total_length += length
            total_processed += total
        print 'Processed %2.2f %% of %d bytes (compressed).  Found %d URLs' %\
            ((total_processed * 100.0 / total_length), total_length, total_urls)
        print "Mime types:"
        for k, v in mime_types.most_common():
            print '%5d %s' % (v, k)

for index in [INDEX1]:
    print 'Processing index: ', index
    process_index(index)
    print 'Done processing index: ', index