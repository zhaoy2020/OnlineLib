import reflex as rx 


class AuthState(rx.State):
    '''The state for the authentication page.'''

    username: str = ""
    password: str = ""

    def login(self):
        '''A fake login function.'''
        print(f"Logging in with username: {self.username} and password: {self.password}")


