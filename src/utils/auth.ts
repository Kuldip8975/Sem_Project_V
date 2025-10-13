// src/utils/auth.ts

export interface User {
    id: string;
    firstname: string;
    lastname: string;
    email: string;
    name?: string;
    role?: string;
  }
  
  export class AuthService {
    private static TOKEN_KEY = 'authToken';
    private static USER_KEY = 'userInfo';
  
    static isAuthenticated(): boolean {
      const token = localStorage.getItem(this.TOKEN_KEY);
      const userInfo = localStorage.getItem(this.USER_KEY);
      
      if (!token || !userInfo) {
        return false;
      }
  
      try {
        JSON.parse(userInfo);
        return true;
      } catch (error) {
        this.logout(); // Clear corrupted data
        return false;
      }
    }
  
    static getToken(): string | null {
      return localStorage.getItem(this.TOKEN_KEY);
    }
  
    static getUser(): User | null {
      const userStr = localStorage.getItem(this.USER_KEY);
      if (!userStr) return null;
      
      try {
        return JSON.parse(userStr);
      } catch (error) {
        this.logout();
        return null;
      }
    }
  
    static setAuth(token: string, user: User): void {
      localStorage.setItem(this.TOKEN_KEY, token);
      localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    }
  
    static logout(): void {
      localStorage.removeItem(this.TOKEN_KEY);
      localStorage.removeItem(this.USER_KEY);
    }
  
    static getUserDisplayName(): string {
      const user = this.getUser();
      if (!user) return 'Guest User';
      
      return user.name || `${user.firstname || ''} ${user.lastname || ''}`.trim() || user.email || 'User';
    }
  }
  
  // Hook for React components
  export const useAuth = () => {
    return {
      isAuthenticated: AuthService.isAuthenticated(),
      user: AuthService.getUser(),
      token: AuthService.getToken(),
      logout: AuthService.logout,
      getUserDisplayName: AuthService.getUserDisplayName
    };
  };