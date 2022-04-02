# 모듈 정리



## requests

* parameter

  * url ( 필수 )

    : url 매개변수는 ruquests.request 객체에 사용되기 위한 URL

  * params ( 선택 사항 )

    : 튜플, 딕셔너리 형식으로 매개변수에 넣으면 양식이 URL 인코딩이 되어 URL에 추가됨

    > url?key=value&key=value1

  * date ( 선택사항 )

    : 튜플 딕셔너리 형식으로 매개변수에 넣으면 양식이 인코딩되어 요청 본문에 추가됨

    > key =value&key1=value1

  * jason ( 선택사항 )

    : JSON 매개변수를 이용하여 요청 본문에 json 형식으로 추가됨

    > { ' key ' : ' value ' , ' key1 ' : ' value1 ' }

  * **kwargs { 선택 사항 }

    : *kwargs는 요청하기 위한 매개변수이며 requests.sessions.Session.request로 연결되어 처리됨

  * return

    : [ PUT, GET, POST, HEAD, PATCH, DELETE, OPTIONS ] 는 기본적으로 requests.modules.Response 객체를 반환함

  ___

  1. PUT

     > requsts.put(url, data=None, **kwargs)

     ```python
     def put (url, data=None, **kwargs):
     
     	[ ... ]
     
     	return request('put', url, data = data, **kwargs
     ```

  2. GET

     > requests.get(url, params=None, **kwargs)

     ```python
     def get(url, params=None, **kwargs):
         [ ... ]
         kwargs.setdefault('allow_redirects', True)
         return request('get', url, params=params, **kwargs)
     ```

  3. POST

     > requests.post(url, data=None, json=None, **kwargs)

     ```python
     def post(url, data=None, json=None, **kwargs):
         [ ... ]
         return request('post', url, data=data, json=json, **kwargs)
     ```

  4.  HEAD

     > requests.head(url, **kwargs)

     ```python
     def head(url, **kwargs):
         [ ... ]
         kwargs.setdefault('allow_redirects', False)
         return request('head', url, **kwargs)
     ```

  5. PATCH

     > requests.patch(url, data=None, **kwargs)

     ```python
     def patch(url, data=None, **kwargs):
         [ ... ]
         return request('patch', url, data=data, **kwargs)
     ```

  6. DELETE

     > requests.delete(url, **kwargs)

     ```python
     def delete(url, **kwargs):
         [ ... ]
         return request('delete', url, **kwargs)
     ```

  7.  OPTIONS

     > requests.options(url, **kwargs)

     ```python
     def options(url, **kwargs):
         [ ... ]
         kwargs.setdefault('allow_redirects', True)
         return request('option', url, **kwargs)
     ```

  ___

  * kwargs 매개변수 종류

    * method(str)

      > method 매개변수는 요청 시 사용될 http 메소드
      >
      > GET 또는 POST 등을 넣으면 됨

      ```python
      >>> r = requests.request(method = 'GET', url = 'https://example.com')
      >>> r
      <Response [200]>
      >>> r = requests.request(method = 'PUT', url = 'http://httpbin.org/put')
      <Resoinse [200]>
      ```

    * url(str)

      > Core Code
      >
      > url 매개변수는 요청하고 싶은 URL을 넣으면 됨

      ```python
        >>> r = requests.request('GET', url='https://example.com')
        <Response [200]>
      ```

    * params( str, dict )

      > Core Code
      >
      > params 매개변수는 요청하는 URL뒤에 GET방식으로 파라미터가 붙음

      ```python
        >>> r = requests.request('GET', url='https://example.com', params={'get1':'value1',   'get2','value2'})
        <Response [200]>
        >>> r.url
        'https://example.com?get=value1&get2=value2'
        >>> r = requests.get("https://www.google.com", params="helloworld")
        >>> r.url
        'https://www.google.com/?helloworld'
      ```

    * data(str, dict)

      > Core Code
      >
      > data 매개변수는 요청될 때 본문에 포함되어 서버로 데이터를 전송함
      >
      > data 매개변수에 dict 또는 문자열 그대로 담아 요청을 할 수 있음

      ```python
        >>> r = requests.request('POST', url='http://httpbin.org/post', data={'post1':'value1', 'post2':'value2'})
        >>> r
        <Response [200]>
        >>> r.request.body
        'post1=value1&post2=value2'
        >>> r = requests.post('http://httpbin.org/post', data="hello post data")
        >>> r.request.body
        'hello post data'
      ```

    * headers( dict )

      > Core Code
      >
      > headers 매개변수는 요청할 때 기본적인 헤더에 추가/ 수정/ 편집하여 서버에 전송

      ```python
       >>> r = requests.request('GET', url='http://httpbin.org/get', headers={'header_test':'test'})
       >>> r.request.headers
        {'User-Agent': 'python-requests/2.25.1', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'header_test': 'test'}
        >>> r.request.headers['header_test']
        'test'
      ```

    * verify ( bool, str )

      > Core Code
      >
      > verify는 서버 TLS 인증서 확인 여부를 제어하기 위해 사용이 됨
      >
      > True, False 또는 certificate의 경로를 넣어주어 사용 가능

      ```python
        >>> import requests
        >>> r = requests.get("<SSL 만료된 URL>")
        Traceback (most recent call last):
          [ ... ]
          File "/home/me2nuk/.local/lib/python3.8/site-packages/requests/adapters.py", line 517, in send
            raise SSLError(e, request=request)
        requests.exceptions.SSLError: HTTPSConnectionPool(host='<SSL 만료된 URL>', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)')))
        >>>
        >>> r = requests.get("<SSL 만료된 URL>", verify=False)
        /home/me2nuk/.local/lib/python3.8/site-packages/urllib3/connectionpool.py:1013: InsecureRequestWarning: Unverified HTTPS request is being made to host '<SSL 만료된 URL>'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings
        warnings.warn(
        >>> r
        <Response [200]>
        >>>
        >>> import certifi
        >>> r = requests.get("https://www.google.com", verify=certifi.where())
        >>> r
        <Response [200]>
      ```

    * json( dict )

      > Core Code
      >
      > json 매개변수는 어떠한 서버에 json 데이터를 전송해야되는 경우 쓰임
      >
      > json 매개변수를 사용하면 요청 헤더에 기본적으로 Content-Type이 application/jso 으로 지정이 된 상태로 요청됨

      ```python
        >>> import requests
        >>> r = requests.get("https://example.com", json={'test1':'jsondata2'})
        >>> r.request.headers
        {'User-Agent': 'python-requests/2.22.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Length': '22', 'Content-Type': 'application/json'}
        >>> r.request.body
        b'{"test1": "jsondata2"}'
        >>> r.request.body.decode()
        '{"test1": "jsondata2"}
      ```

    ---

  * Response

    > Class requests.modules.Response
    >
    > `Response`는 HTTP 요청에 대한 서버의 응답을 포함한 객체

    ```python
    >>> import requests
    >>> r = requests.get("https:/example.com")
    ```

    * r.text

      > Core Code
      >
      > text는 요청/응답 본문을 자동으로 디코드시킨 값을 str타입으로 반환

      ```python
        >>> r.text
        '<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>[ ... ]v>\n</body>\n</html>\n'
        >>> print(r.text)
        <!doctype html>
        <html>
        <head>
            <title>Example Domain</title>
      
            <meta charset="utf-8" />
            <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style type="text/css">
            body {
                background-color: #f0f0f2;
                margin: 0;
                padding: 0;
            [ ... ]
        </body>
        </html>
      ```

    * r.content

      > Core Code
      >
      > content는 요청/응답 본문을 byte 타입으로 반환됨

      ```python
        >>> r.content
        b'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title> [ ... ] </body>\n</html>\n'
        >>> type(r.content)
        <class 'bytes'>
        >>> print(r.content.decode())
        <!doctype html>
        <html>
        <head>
            <title>Example Domain</title>
      
            <meta charset="utf-8" />
            <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style type="text/css">
            body {
                background-color: #f0f0f2;
                margin: 0;
                padding: 0;
                font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
      
            }
            div {
                width: 600px;
            [ ... ]
        </body>
        </html>
        >>>
      ```

    * r.json()

      > Core Code
      >
      >  `json(self, **kwargs)`
      >
      > json() 는 요청/응답 본문을 json 형식으로 디코딩하여 반환됨
      >
      > 만약 올바른 json 형식이 아닌 경우 에러를 반환됨

      ```python
        >>> import requests
        >>> r = requests.get("https://example.com")
        >>> r.json()
        Traceback (most recent call last):
          File "<[ ... ] "
            raise JSONDecodeError("Expecting value", s, err.value) from None
        json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
        >>> r = requests.get("https://api.github.com/events")
        >>> r.json()
        [{'id': '16070464781', 'type': 'ForkEvent', 'actor': [ ... ] 'at': '2021-04-24T10:30:31Z'}]
        >>> type(r.json())
        <class 'list'>
      ```

    * r.status_code

      > Core Code
      >
      > `status_code` 는 http 응답 코드를 나타냄
      >
      > 요청에 성공한 경우 일반적으로 200을 반환

    * r.url

      > Core Code
      >
      > `url`은 요청한 뒤 응답의 최종 URL을 반환
      >
      > URL redirection이 되는 경우에도 리다이렉션이 된 최종 URL을 출력
      >
      > `http://127.0.0.1:8080/redirect`->`/redirect_test`

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.url
        'https://example.com'
        >>>
        >>> r = requests.get("http://127.0.0.1:8080/redirect")
        >>> r.url
        'http://127.0.0.1:8080/redirect_test'
      ```

    * r.history

      > Core Code
      >
      > history는 모든 리다이렉션 응답은 가장 오래된 요청에서 최근 요청 순으로 Response 개체 목록을 반환함
      >
      > 이해가 더 잘되기 위해 예시로 로컬에서 Flask으로 여러번 리다이렉션을 반복하여 테스트
      >
      >  `127.0.0.1:8080/redirect/n Code`

      ```python
        from flask import Flask, redirect
      
        app = Flask(__name__)
      
        @app.route('/redirect/<int:n>')
        def redirects(n):
            return (redirect(f'/redirect/{n-1}', code=302) if n >= 1 else 'redirect TEST end')
      
        app.run('127.0.0.1', 8080)
      ```

    * r.links

      > Core Code
      >
      > links는 요청/응답 헤더의 link를 피싱한 결과를 반환합니다.
      >
      > 만약 존재하지 않는 경우 { } 빈 딕셔너리를 반환함

      ```python
        >>> r = requests.get("https://api.github.com/users/kennethreitz/repos?page=1&per_page=10")
        >>> r.headers['link']
        '<https://api.github.com/user/119893/repos?page=2&per_page=10>; rel="next", <https://api.github.com/user/119893/repos?page=5&per_page=10>; rel="last"'
        >>> r.links
        {'next': {'url': 'https://api.github.com/user/119893/repos?page=2&per_page=10', 'rel': 'next'}, 'last': {'url': 'https://api.github.com/user/119893/repos?page=5&per_page=10', 'rel': 'last'}}
        >>>
        >>>
        >>> r = requests.get("https://example.com")
        >>> r.headers['link']
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/lib/python3/dist-packages/requests/structures.py", line 52, in __getitem__
            return self._store[key.lower()][1]
        KeyError: 'link'
        >>> r.links
        {}
      ```

    * r.headers

      > Core Code
      >
      > `headers`는 요청한 뒤 응답 헤더를 반환함

      ```python
       >>> r = requests.get("https://example.com")
        >>> r.headers
        {'Content-Encoding': 'gzip', 'Age': '330304', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Fri, 30 Apr 2021 13:58:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Fri, 07 May 2021 13:58:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (sab/56BC)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648'}
      ```

    * r.cookies

      > Core Code
      >
      > `cookies`는 요청한 뒤 응답 헤더에 있는 쿠키를 편하게 보여줌

      ```python
        >>> r = requests.get("https://google.com")
        >>> r.headers['set-cookie']
        '1P_JAR=2021-04-30-14; expires=Sun, 30-May-2021 14:02:07 GMT; path=/; domain=.google.com; Secure, NID=214=V54rp0jqnDG7IFhEI8bUU1DhK8FERCtCfFYzPlPNdCgFLZTmwxQpUhUEzc5xtK_p_4BByikl28WX7558B2WWmY7iJPMMPiMmhnwvZbftcazRwyPLjDjgaA_3GBKRMkipp7qD0ONumogYbm9tbjaRCYjp08qNxfeDjOIgLiGSdaU; expires=Sat, 30-Oct-2021 14:02:07 GMT; path=/; domain=.google.com; HttpOnly'
        >>>
        >>> r.cookies
        <RequestsCookieJar[Cookie(version=0, name='1P_JAR', value='2021-04-30-14', port=None, port_specified=False, domain='.google.com', domain_specified=True, domain_initial_dot=True, path='/', path_specified=True, secure=True, expires=1622383327, discard=False, comment=None, comment_url=None, rest={}, rfc2109=False), Cookie(version=0, name='NID', value='214=V54rp0jqnDG7IFhEI8bUU1DhK8FERCtCfFYzPlPNdCgFLZTmwxQpUhUEzc5xtK_p_4BByikl28WX7558B2WWmY7iJPMMPiMmhnwvZbftcazRwyPLjDjgaA_3GBKRMkipp7qD0ONumogYbm9tbjaRCYjp08qNxfeDjOIgLiGSdaU', port=None, port_specified=False, domain='.google.com', domain_specified=True, domain_initial_dot=True, path='/', path_specified=True, secure=False, expires=1635602527, discard=False, comment=None, comment_url=None, rest={'HttpOnly': None}, rfc2109=False)]>
        >>>
        >>> for key,value in r.cookies.items():
        ...     print(key, value)
        ...
        1P_JAR 2021-04-30-14
        NID 214=V54rp0jqnDG7IFhEI8bUU1DhK8FERCtCfFYzPlPNdCgFLZTmwxQpUhUEzc5xtK_p_4BByikl28WX7558B2WWmY7iJPMMPiMmhnwvZbftcazRwyPLjDjgaA_3GBKRMkipp7qD0ONumogYbm9tbjaRCYjp08qNxfeDjOIgLiGSdaU
      ```

    * r.connection

      > Core Code
      >
      > `connection`은 requests 모듈 내부에서 요청을 한 다음 response를 처리하는 과정인 context가 들어있음

      ```python
        >>> import requests
        >>> r = requests.get("Https://www.google.com")
        >>> r.connection
        <requests.adapters.HTTPAdapter object at 0x7f8dea56f6d0>
        >>> type(r.connection)
        <class 'requests.adapters.HTTPAdapter'>
        >>> r.connection.config
        {}
        >>> r.connection.proxy_manager
        {}
        >>> r.connection.max_retries
        Retry(total=0, connect=None, read=False, redirect=None, status=None)
        >>> r.connection._pool_connections
        10
        >>> r.connection._pool_maxsize
        10
        >>> r.connection._pool_block
        False
      ```

    * r.elapsed

      > Core Code
      >
      > `elapsed`는 요청을 보낸 후 응답이 도착할 때가지의 경과한 시간을 datetime.timedelta 객체로 반환함

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.elapsed
        datetime.timedelta(microseconds=643703)
        >>> r.elapsed.microseconds
        643703
        >>> print(r.elapsed)
        0:00:00.643703
      ```

    * r.ok

      > Core Code
      >
      > `ok`는 요청/응답 코드가 200이면 True 아니면 False를 반환하
      >
      > `raise_for_status()`를 이용하여 예외처리로 True, False를 구별함

      ```python
        @property
        def ok(self):
            [ ... ]
            try:
                self.raise_for_status()
            except HTTPError:
                return False
            return True
      ```

    * r.reason

      > Core Code
      >
      > `reason`는 요청/응답 http 상태 코드의 텍스트를 출력함

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.ok
        True
        >>> r.status_code
        200
        >>> r.reason
        'OK'
        >>>
        >>> r = requests.get("https://example.com/a/")
        >>> r.ok
        False
        >>> r.status_code
        404
        >>> r.reason
      ```

    * r.raise_for_status()

      > Core Code
      >
      > `raise_for_status()`는 요청/응답 코드가 200이 아니면 예외를 발생시킴

      ```python
        >>> r = requests.put("https://google.com")
        >>> r.raise_for_status()
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/lib/python3/dist-packages/requests/models.py", line 940, in raise_for_status
            raise HTTPError(http_error_msg, response=self)
        requests.exceptions.HTTPError: 405 Client Error: Method Not Allowed for url: https://google.com/
        >>> r = requests.get("https://example.com")
        >>> r.raise_for_status()
        >>> type(r.raise_for_status())
        <class 'NoneType'>
        >>> r = requests.get("https://example.com/a")
        >>>
        >>> r.raise_for_status()
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/lib/python3/dist-packages/requests/models.py", line 940, in raise_for_status
            raise HTTPError(http_error_msg, response=self)
        requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://example.com/a
      ```

    * r.encoding

      > Core Code
      >
      > `encoding`는 요청/응답 헤더를 이용하여 데이터의 인코딩 방식을 추측하여 반환함

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.encoding
        'UTF-8'
        >>> r.headers
        { [ ... ] 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Thu, 29 Apr 2021 14:21 [ ... ] IT', 'Content-Length': '648'}
      r.apparent_encoding
      ```

    * r.close()

      > Core Code
      >
      > close()는 서버와의 연결을 닫음

      ```python
        >>> r = requests.get("https://example.com")
        >>> r
        <Response [200]>
        >>> r.close
        >>> r.close()
      ```

    * r.request

      > Core Code
      >
      > `request`는 Preparedrequest클래스를 반환하며 요청시에 사용했던 정보들을 확인할 수 있음

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.request
        <PreparedRequest [GET]>
        >>> dir(r.request)
        ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_body_position', '_cookies', '_encode_files', '_encode_params', '_get_idna_encoded_host', 'body', 'copy', 'deregister_hook', 'headers', 'hooks', 'method', 'path_url', 'prepare', 'prepare_auth', 'prepare_body', 'prepare_content_length', 'prepare_cookies', 'prepare_headers', 'prepare_hooks', 'prepare_method', 'prepare_url', 'register_hook', 'url']
      r.request.method
      ```

    * r.request.method

      > Core Code
      >
      > HTTP/HTTPS 요청 메서드를 나타냄

      ```python
        >>> import requests
        >>> r = requests.get("https://www.google.com")
        >>> r.request.method
        'GET'
        >>> r = requests.post("https://www.google.com")
        >>> r.request.method
        'POST'
      ```

    * r.request.path_url

      > Core Code
      >
      > 요청한 url의 경로를 나타냄

      ```python
        >>> import requests
        >>> r = requests.get("https://www.google.com")
        >>> r.request.path_url
        '/'
        >>> r = requests.get("https://www.google.com/test/path")
        >>> r.request.path_url
        '/test/path'
      ```

    * r.request.url

      > Core Code
      >
      > 요청한 URL 전체를 나타냄

      ```python
        >>> import requests
        >>> r = requests.get("https://www.google.com")
        >>> r.request.url
        'https://www.google.com'
        >>> r = requests.get("https://www.google.com/test/aps/ds/fasfas")
        'https://www.google.com/test/aps/ds/fasfas'
      ```

    * r.request.headers

      > Core Code
      >
      > `request.headers`는 요청할 때 사용된 헤더를 dict 타입으로 반환함

      ```python
        >>> r = requests.get("https://example.com")
        >>> r.request
        <PreparedRequest [GET]>
        >>>
        >>> r.request.headers
        {'User-Agent': 'python-requests/2.22.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
        >>> r.request.headers['User-Agent']
        'python-requests/2.22.0'
      ```

    * r.request._cookies

      > Core Code
      >
      > `request._cookies`는 요청할 때 사용된 쿠키 내용을 dict 타입으로 반환됨

      ```python
        >>> r = requests.get("https://example.com", cookies={'cookie1':'cookie_value'})
        >>> r.request
        <PreparedRequest [GET]>
        >>>
        >>> r.request._cookies
        >>> r.request._cookies['cookie1']
        'cookie_value'
        >>> r.request._cookies.get_dict()
        {'cookie1': 'cookie_value'}
      ```

    * r.raw

      > Core Code
      >
      > 서버에서 원시 소켓 응답을 받기 위해 `r.raw.*`를 사용하기 위해서는 요청 시 `stream=True`를 추가해줘야 함

      ```python
        >>> r = requests.get("https://example.com", stream=True)
        >>> r.raw
        <urllib3.response.HTTPResponse object at 0x7f53bba651f0>
        >>> dir(r.raw)
        ['CONTENT_DECODERS', 'DECODER_ERROR_CLASSES', 'REDIRECT_STATUSES', '__abstractmethods__', '__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_abc_impl', '_body', '_checkClosed', '_checkReadable', '_checkSeekable', '_checkWritable', '_connection', '_decode', '_decoder', '_error_catcher', '_flush_decoder', '_fp', '_fp_bytes_read', '_handle_chunk', '_init_decoder', '_init_length', '_original_response', '_pool', '_request_url', '_update_chunk_length', 'auto_close', 'chunk_left', 'chunked', 'close', 'closed', 'connection', 'data', 'decode_content', 'enforce_content_length', 'fileno', 'flush', 'from_httplib', 'get_redirect_location', 'getheader', 'getheaders', 'geturl', 'headers', 'info', 'isatty', 'isclosed', 'length_remaining', 'msg', 'read', 'read_chunked', 'readable', 'readinto', 'readline', 'readlines', 'reason', 'release_conn', 'retries', 'seek', 'seekable', 'status', 'stream', 'strict', 'supports_chunked_reads', 'tell', 'truncate', 'version', 'writable', 'writelines']
      
      ```

    * r.raw.read()

      > `read(self, amt=None, decode_content=None, cache_content=False)`
      >
      > `r.raw.read()`함수를 이용하여 응답 본문 컨텐츠를 원하는 만큼 인코딩 된 값을
      >
      > 출력할 수 있음
      >
      > 해당 기능은 open.read 함수와 유사

      ```python
        >>> r = requests.get("https://example.com" ,stream = True)
        >>> r.raw
        <urllib3.response.HTTPResponse object at 0x7f8f692dc8b0>
        >>> r.raw.read()
        b'\x1f\x8b\x08\x00\xc2\x15\xa8]\x00\x03}TMs\xdb \x10\xbd\xfbWl\xd5K2#$\'i\x1a\x8f-i\xfa\x99i\x0fi\x0fi\x0f=\x12\xb1\xb2\x98\x08P\x01\xc9\xf6t\xf2\xdf\xbbB\x8e#7\x99\x9a\x91[ ... ]d0x\x11\x10\xb34\x88\x93\xa5{\xa9\xd2\xf1A\xfb\x0b(\xeb|o\xe8\x04\x00\x00'
        >>> r.raw.read(10)
        b''
        >>> r = requests.get("https://example.com", stream=True)
        >>>
        >>> r.raw.read(10)
        b'\x1f\x8b\x08\x00\xc2\x15\xa8]\x00\x03'
        >>> r.raw.read(10)
        b'}TMs\xdb \x10\xbd\xfbW'
      ```

      

