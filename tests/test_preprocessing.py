import pytest

from job_simulation.preprocessing import remove_punctuation, remove_emails, preprocess_text


@pytest.mark.parametrize(
    'text, output',
    [
        ('..!!,', ''),
        ('///', ''),
        ('wells_fargo', 'wells_fargo'),
        ('XX/XX/2020', 'XXXX2020'),
        ('te,st', 'test'),
        ('', ''),
        ('hello', 'hello')
    ]
)
def test_remove_punctuation(text, output):
    assert output == remove_punctuation(text)


@pytest.mark.parametrize(
    'text, output',
    [
        ('smith@gmail.com', ' '),
        ('hello smith@gmail.com', ' '),
        ('@smith', ' '),
        ('a@b', ' '),
        ('a@', ' '),
    ]
)
def test_remove_emails(text, output):
    assert output == remove_emails(text)


@pytest.mark.parametrize(
    'text, output',
    [
        ('Hello, my name is Alex, alex@gmail.com, 5:09, 07-12-2017',
         'hello my name is alex alexgmailcom xxxxxxxx xxxxxxxxxxxx'),
    ]
)
def test_preprocess_text(text, output):
    assert output == preprocess_text(text)