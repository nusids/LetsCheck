# LetsCheck Website

The LetsCheck website is a static HTML site. It is currently hosted through Caddy on A6000 workstation in IDS office. 

## Deployment on NUS VM
The public-facing [LetsCheck website](https://letscheck.nus.edu.sg/general.html) used to be hosted on NUS VM with Apache web server, however it is currently not maintained.

The details of the NUS VM are:
```
Server name: lxwebprod05012.res.nus.edu.sg
IP: 137.132.174.114
```

The SSL certificates are managed through nCertRequest. 

For any issues regarding NUS VM or certificate, please contact NUS IT UNIX System Administrators <NUSITUNIXSystemAdministrators@nus.edu.sg> with the information above.

## Notes
- Ensure there are no inline scripts in html files; it is forbidden by the NUS VM [content security policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP). Put JS in the corresponding `.js` files instead.
