# aws 설정

CLI를 설치하지 않고 따로 credentials를 지정해 줘야 한다.
AWS에서는 해당 credentials을 바라보는 경로가 있다.

```bash
vim ~/.aws/credentials
```

안에 들어가야할 내용은 다음과 같다.

```md
[default]
aws_access_key_id = {ACCESS KEY}
aws_secret_access_key = {SECRET ACCESS KEY}
```
