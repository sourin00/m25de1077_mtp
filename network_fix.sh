# 1. Dump the certs macOS trusts (includes your proxy's root CA) into one bundle
security find-certificate -a -p /Library/Keychains/System.keychain > ~/corp-ca.pem
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >> ~/corp-ca.pem

# 2. Point pip, requests/HF, and raw ssl at it
export PIP_CERT=~/corp-ca.pem            # pip
export REQUESTS_CA_BUNDLE=~/corp-ca.pem  # huggingface_hub, transformers, requests
export SSL_CERT_FILE=~/corp-ca.pem       # raw ssl + httpx (openai / anthropic SDKs)
export CURL_CA_BUNDLE=~/corp-ca.pem      # curl-based tools

python -c "import urllib.request, os; urllib.request.urlopen('https://pypi.org'); print('TLS OK')"